#include "ds41/store.hpp"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
namespace ds41 {
Placement load_placement(const std::string& path) {
    std::ifstream f(path);if(!f)throw std::runtime_error("cannot open placement: "+path);
    Placement p;std::unordered_map<ExpertId,bool> selected;std::string line;
    while(std::getline(f,line)) {
        if(line.empty() || line[0]=='#')continue;
        std::istringstream in(line);std::string op;in>>op;
        if(op=="vram" || op=="host"){
            ExpertId id;in>>id;
            if(!in || id>=expert_count || !selected.emplace(id,true).second)throw std::runtime_error("invalid/overlapping placement expert");
            (op=="vram"?p.vram_pins:p.host_residents).push_back(id);
        }else if(op=="pools"){
            std::string mode;in>>mode;if(mode!="hard" && mode!="soft")throw std::runtime_error("invalid pool mode");p.hard_layer_budgets=mode=="hard";
        }else if(op=="budget" || op=="host_budget"){
            unsigned layer;uint64_t slots;in>>layer>>slots;
            if(!in || layer>=43 || slots>(layer<40?384u:128u))throw std::runtime_error("invalid layer slot budget");
            (op=="budget"?p.dynamic_slots_per_layer:p.host_slots_per_layer)[layer]=slots;
        }else if(op=="recent" || op=="decay"){
            uint64_t n;in>>n;if(!in || n>UINT32_MAX)throw std::runtime_error("invalid policy interval");
            (op=="recent"?p.recent_window:p.decay_interval)=n;
        }else if(op=="lfu"){
            std::string mode;in>>mode;if(mode!="distinct" && mode!="multiplicity")throw std::runtime_error("invalid LFU counting mode");
            p.distinct_lfu=mode=="distinct";
        }else throw std::runtime_error("unknown placement directive");
        std::string tail;if(in>>tail)throw std::runtime_error("trailing placement tokens");
    }
    return p;
}
std::vector<Record> load_records(const std::string& path) {
    std::ifstream f(path);if(!f)throw std::runtime_error("cannot open manifest: "+path);
    std::vector<Record> records;std::unordered_map<ExpertId,size_t> ids;
    std::string line;
    while(std::getline(f,line)){
        if(line.empty() || line[0]=='#')continue;
        std::istringstream s(line);char tag;ExpertId id;s>>tag>>id;
        if(tag=='R'){
            Record r;r.id=id;
            s>>r.bytes>>r.gate.offset>>r.gate.bytes>>r.gate_scale.offset>>r.gate_scale.bytes
             >>r.up.offset>>r.up.bytes>>r.up_scale.offset>>r.up_scale.bytes>>r.down.offset>>r.down.bytes>>r.down_scale.offset>>r.down_scale.bytes;
            if(!s || id>=expert_count || !ids.emplace(id,records.size()).second)throw std::runtime_error("bad record line");
            records.push_back(std::move(r));
        }else if(tag=='E'){
            Extent e;s>>std::quoted(e.path)>>e.file_offset>>e.bytes>>e.dst_offset>>e.required_bytes>>e.expected_file_bytes;
            if(!s || !ids.count(id))throw std::runtime_error("bad extent line");
            records[ids.at(id)].extents.push_back(std::move(e));
        }else throw std::runtime_error("unknown manifest line");
    }
    return records;
}
}
