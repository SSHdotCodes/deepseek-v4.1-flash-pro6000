#include "ds41/spark.h"
#include <algorithm>
#include <cmath>
#include <cstring>
extern "C" void ds41_spark_policy_default(ds41_spark_policy_config*c){
    if(!c)return;
    *c={};c->max_width=6;c->min_samples=32;c->ewma=.1;c->prior_acceptance=.5;
    double draft[]={0,2.632,2.966,3.067,3.062,3.286},base[]={14,15.2,16.5,17.8,19.1,20.4};
    for(int i=0;i<6;++i){c->draft_ms[i]=draft[i];c->base_verify_ms[i]=base[i];}
    c->host_gbps=37.37;c->nvme_gbps=9.23;c->expert_bytes=18800640;c->margin=.02;
}
extern "C" void ds41_spark_policy_reset(ds41_spark_policy*s,const ds41_spark_policy_config*c){
    if(!s||!c)return;
    *s={};for(int i=0;i<5;++i)s->acceptance[i]=c->prior_acceptance;
}
extern "C" int ds41_spark_policy_observe(ds41_spark_policy*s,const ds41_spark_policy_config*c,int width,int accepted,double ms,double host,double nvme){
    if(!s||!c||width<1||width>6||accepted<0||accepted>=width||!std::isfinite(ms)||ms<0||!std::isfinite(host)||host<0||!std::isfinite(nvme)||nvme<0||c->ewma<=0||c->ewma>1)return -1;
    int i=width-1;double a=s->steps[i]?c->ewma:1.;++s->steps[i];s->measured_ms[i]+=a*(ms-s->measured_ms[i]);s->host_misses[i]+=a*(host-s->host_misses[i]);s->nvme_misses[i]+=a*(nvme-s->nvme_misses[i]);
    for(int k=0;k<width-1&&k<=accepted;++k){++s->reached[k];s->accepted[k]+=k<accepted;s->acceptance[k]+=c->ewma*((k<accepted?1.:0.)-s->acceptance[k]);}return 0;
}
extern "C" int ds41_spark_policy_choose_ex(const ds41_spark_policy*s,const ds41_spark_policy_config*c,int period,int explore_width,int remaining){
    if(remaining<=0)return 0;
    if(!s||!c||c->max_width<1||c->max_width>6||c->host_gbps<=0||c->nvme_gbps<=0)return 1;
    int cap=std::min(c->max_width,std::min(6,remaining));
    uint64_t total=0;for(int i=0;i<6;++i)total+=s->steps[i];
    if(period>0&&(total+1)%uint64_t(period)==0)return std::clamp(explore_width,1,cap);
    int best=1;double rate=0,survival=1,emitted=1;
    for(int w=1;w<=cap;++w){if(w>1){survival*=std::clamp(s->acceptance[w-2],0.,1.);emitted+=survival;}
        int i=w-1;double cost=c->base_verify_ms[i];
        if(s->steps[i]>=uint64_t(std::max(1,c->min_samples)))cost=s->measured_ms[i];
        else{double factor=double(w);double host=s->steps[i]?s->host_misses[i]:s->host_misses[0]*factor,nvme=s->steps[i]?s->nvme_misses[i]:s->nvme_misses[0]*factor;
            cost+=c->expert_bytes/1e6*(host/c->host_gbps+nvme/c->nvme_gbps);}
        cost+=c->draft_ms[i];if(!(cost>0)||!std::isfinite(cost))continue;double candidate=emitted/cost;
        if(candidate>rate*(1+c->margin)){rate=candidate;best=w;}
    }return best;
}

extern "C" int ds41_spark_policy_choose(const ds41_spark_policy*s,const ds41_spark_policy_config*c){return ds41_spark_policy_choose_ex(s,c,16,3,6);}
