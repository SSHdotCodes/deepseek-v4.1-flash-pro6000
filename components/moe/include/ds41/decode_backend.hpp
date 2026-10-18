#pragma once
#include "ds41/kernels.cuh"
#include <string>
namespace ds41 {
// Resolve once before capture. Keep this object alive through graph destruction.
// The optional drop-in never changes table/job layout or reduction order.
class DecodeBackend {
public:
    struct Config {bool enable_xk=false;std::string xk_library;};
    DecodeBackend();
    explicit DecodeBackend(const Config& config);
    ~DecodeBackend();
    DecodeBackend(const DecodeBackend&)=delete;
    DecodeBackend& operator=(const DecodeBackend&)=delete;
    bool uses_xk()const noexcept{return grouped_!=nullptr;}
    cudaError_t launch(const DeviceExpert* table,const GemmJob* jobs,int count,
                      Projection projection,int n,int k,cudaStream_t stream)const;
private:
    using Grouped=int(*)(const DeviceExpert*,const GemmJob*,int,int,int,int,void*);
    void* module_=nullptr;
    Grouped grouped_=nullptr;
};
}
