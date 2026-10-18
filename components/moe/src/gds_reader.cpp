#include "ds41/gds_reader.hpp"
#include <cufile.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <stdexcept>

namespace ds41 {
struct GdsReader::Api {
    decltype(&cuFileDriverOpen) driver_open;
    decltype(&cuFileDriverClose) driver_close;
    decltype(&cuFileHandleRegister) handle_register;
    decltype(&cuFileHandleDeregister) handle_deregister;
    decltype(&cuFileBufRegister) buf_register;
    decltype(&cuFileBufDeregister) buf_deregister;
    decltype(&cuFileRead) read;
};
namespace {
template<class F> F sym(void* lib, const char* name) {
    void* p = dlsym(lib, name);
    if (!p) throw std::runtime_error(std::string("cuFile symbol missing: ") + name);
    return reinterpret_cast<F>(p);
}
void check(CUfileError_t e, const char* what) {
    if (e.err != CU_FILE_SUCCESS) throw std::runtime_error(std::string(what) + " failed: cuFile error " + std::to_string(int(e.err)));
}
}
GdsReader::GdsReader(void* base, size_t bytes, size_t piece_bytes, unsigned threads)
    : base_(base), bytes_(bytes), piece_(piece_bytes) {
    if (!base || !bytes || !piece_bytes || piece_bytes % 4096 || !threads) throw std::invalid_argument("GDS reader geometry");
    lib_ = dlopen("libcufile.so.0", RTLD_NOW | RTLD_LOCAL);
    if (!lib_) throw std::runtime_error(std::string("libcufile: ") + dlerror());
    api_ = new Api{sym<decltype(Api::driver_open)>(lib_, "cuFileDriverOpen"), sym<decltype(Api::driver_close)>(lib_, "cuFileDriverClose"),
                   sym<decltype(Api::handle_register)>(lib_, "cuFileHandleRegister"), sym<decltype(Api::handle_deregister)>(lib_, "cuFileHandleDeregister"),
                   sym<decltype(Api::buf_register)>(lib_, "cuFileBufRegister"), sym<decltype(Api::buf_deregister)>(lib_, "cuFileBufDeregister"),
                   sym<decltype(Api::read)>(lib_, "cuFileRead")};
    check(api_->driver_open(), "cuFileDriverOpen");
    try { check(api_->buf_register(base_, bytes_, 0), "cuFileBufRegister"); }
    catch (...) { api_->driver_close(); throw; }
    for (unsigned i = 0; i < threads; ++i) workers_.emplace_back([this] { worker(); });
}
GdsReader::~GdsReader() {
    { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
    work_cv_.notify_all();
    for (auto& t : workers_) t.join();
    if (api_) {
        api_->buf_deregister(base_);
        for (auto& [path, f] : files_) { if (f.handle) api_->handle_deregister(static_cast<CUfileHandle_t>(f.handle)); if (f.fd >= 0) close(f.fd); }
        api_->driver_close();
        delete api_;
    }
    if (lib_) dlclose(lib_);
}
void* GdsReader::handle(const std::string& path) {
    auto& f = files_[path];
    if (f.handle) return f.handle;
    f.fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    if (f.fd < 0) throw std::runtime_error("GDS open " + path);
    CUfileDescr_t d{};
    d.handle.fd = f.fd;
    d.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    CUfileHandle_t h{};
    check(api_->handle_register(&h, &d), "cuFileHandleRegister");
    f.handle = h;
    return h;
}
void GdsReader::worker() {
    for (;;) {
        Piece p;
        void* fh;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            work_cv_.wait(lock, [this] { return stopping_ || next_ < queue_.size(); });
            if (stopping_) return;
            p = queue_[next_++];
            fh = files_.at(p.extent->path).handle;
        }
        std::string failure;
        ssize_t r = api_->read(static_cast<CUfileHandle_t>(fh), base_, p.bytes, off_t(p.file_offset), off_t(p.device_offset));
        if (r < 0 || uint64_t(r) < p.required) failure = "GDS read " + p.extent->path + " returned " + std::to_string(r);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!failure.empty() && error_.empty()) error_ = failure;
            --remaining_[p.record];
        }
        done_cv_.notify_all();
    }
}
void GdsReader::read(const std::vector<const Record*>& records, const std::vector<size_t>& offsets,
                     const std::function<void(size_t)>& ready) {
    if (records.size() != offsets.size()) throw std::invalid_argument("GDS read geometry");
    std::vector<Piece> pieces;
    std::vector<size_t> counts(records.size());
    for (size_t j = 0; j < records.size(); ++j)
        for (const auto& e : records[j]->extents) {
            handle(e.path);
            const uint64_t required = e.required_bytes ? e.required_bytes : e.bytes;
            for (uint64_t at = 0; at < e.bytes; at += piece_) {
                const uint64_t n = std::min<uint64_t>(piece_, e.bytes - at);
                const uint64_t need = at >= required ? 0 : std::min<uint64_t>(n, required - at);
                const size_t device = offsets[j] + e.dst_offset + at;
                if (device + n > bytes_) throw std::out_of_range("GDS destination outside registered arena");
                pieces.push_back({j, &e, e.file_offset + at, n, need, device});
                ++counts[j];
            }
        }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (next_ != queue_.size()) throw std::logic_error("concurrent GDS reads");
        queue_ = std::move(pieces); next_ = 0; remaining_ = counts; error_.clear();
    }
    work_cv_.notify_all();
    std::string failure;
    for (size_t j = 0; j < records.size(); ++j) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            done_cv_.wait(lock, [&] { return remaining_[j] == 0 || !error_.empty(); });
            if (!error_.empty()) { failure = error_; break; }
        }
        ready(j);
    }
    // Every submitted piece must finish before the caller may reuse or publish slots.
    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [&] { return std::all_of(remaining_.begin(), remaining_.end(), [](size_t n) { return n == 0; }); });
    queue_.clear(); next_ = 0;
    if (!failure.empty()) throw std::runtime_error(failure);
}
} // namespace ds41
