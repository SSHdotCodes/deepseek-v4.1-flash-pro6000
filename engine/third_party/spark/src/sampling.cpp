#include "ds41/spark.h"
#include "ds41/spark_rng.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <string>
#include <stdexcept>
thread_local std::string spark_error;
extern "C" const char* ds41_spark_error(){return spark_error.c_str();}
namespace {
double validate(const double* p,int n){
    if(!p||n<1)throw std::invalid_argument("null/empty probability row");
    double z=0;for(int i=0;i<n;++i){if(!std::isfinite(p[i])||p[i]<0)throw std::invalid_argument("invalid probability");z+=p[i];}
    if(std::abs(z-1)>1e-8)throw std::invalid_argument("probabilities must be normalized");
    return z;
}
int draw(const double* p,int n,double u){double z=0;for(int i=0;i<n;++i)z+=p[i];
    if(!(z>0))throw std::runtime_error("empty residual support");
    double t=u*z,c=0;int last=-1;for(int i=0;i<n;++i){c+=p[i];if(p[i]>0)last=i;if(t<c)return i;}return last;
}
int best(const double*p,int n){return int(std::max_element(p,p+n)-p);}
}
extern "C" uint32_t ds41_spark_rng_word(uint64_t s,uint64_t p,uint32_t u,uint64_t d){if(d/4>UINT32_MAX||u>3){spark_error="Philox purpose/draw counter out of range";return 0;}return spark_word(s,p,u,d);}
extern "C" double ds41_spark_uniform(uint64_t s,uint64_t p,uint32_t u,uint64_t d){if(d/4>UINT32_MAX||u>3){spark_error="Philox purpose/draw counter out of range";return NAN;}return spark_uniform(s,p,u,d);}
extern "C" int ds41_spark_sample(const double*p,int n,uint64_t s,uint64_t pos,uint32_t purpose,int32_t*out){try{
    if(!out||purpose>3)throw std::invalid_argument("sample arguments");
    validate(p,n);*out=draw(p,n,spark_uniform(s,pos,purpose));return 0;
}catch(const std::exception&e){spark_error=e.what();return -1;}}
extern "C" int ds41_spark_probabilities(const float*x,double*p,int n,double t,double top){try{
    if(!x||!p||n<1||!std::isfinite(t)||t<0||!std::isfinite(top)||top<=0||top>1)throw std::invalid_argument("temperature/top_p/logits");
    int b=0;for(int i=0;i<n;++i){if(!std::isfinite(x[i]))throw std::invalid_argument("nonfinite logits");if(x[i]>x[b])b=i;}
    std::fill(p,p+n,0);if(t==0){p[b]=1;return 0;}
    double z=0;for(int i=0;i<n;++i){p[i]=std::exp((double(x[i])-x[b])/t);z+=p[i];}for(int i=0;i<n;++i)p[i]/=z;
    if(top<1){std::vector<int>idx(n);std::iota(idx.begin(),idx.end(),0);std::stable_sort(idx.begin(),idx.end(),[&](int a,int b){return p[a]>p[b];});
        double mass=0;int k=0;do{mass+=p[idx[k++]];}while(k<n&&mass<top);for(int i=k;i<n;++i)p[idx[i]]=0;for(int i=0;i<n;++i)p[i]/=mass;}
    return 0;
}catch(const std::exception&e){spark_error=e.what();return -1;}}
static int verify_impl(const int32_t*d,const double*q,const double*p,int count,int n,int greedy,uint64_t seed,uint64_t pos,ds41_spark_result*out,bool v2){try{
    if(!out||count<0||count>5||n<1||!p||(count&&(!d||(!q&&!greedy)))||(!v2&&pos>UINT64_MAX-5))throw std::invalid_argument("verify arguments");
    auto uniform=[&](int slot,uint32_t purpose){return v2?spark_uniform_v2(seed,pos,slot,purpose):spark_uniform(seed,pos+slot,purpose);};
    ds41_spark_result r{};for(int k=0;k<=count;++k)validate(p+size_t(k)*n,n);
    for(int k=0;k<count;++k){if(d[k]<0||d[k]>=n)throw std::invalid_argument("draft id");const double*pk=p+size_t(k)*n;const double*qk=q?q+size_t(k)*n:nullptr;
        bool accept;
        if(greedy)accept=d[k]==best(pk,n);
        else{validate(qk,n);if(qk[d[k]]<=0)throw std::invalid_argument("draft impossible under logged q");
            accept=uniform(k,DS41_SPARK_ACCEPT)<std::min(1.,pk[d[k]]/qk[d[k]]);}
        if(accept){r.tokens[r.emitted++]=d[k];++r.accepted_drafts;}
        else{if(greedy)r.tokens[r.emitted++]=best(pk,n);else{std::vector<double>res(n);for(int i=0;i<n;++i)res[i]=std::max(0.,pk[i]-qk[i]);r.tokens[r.emitted++]=draw(res.data(),n,uniform(k,DS41_SPARK_RESIDUAL));}r.rejected=1;break;}
    }
    if(!r.rejected){const double*last=p+size_t(count)*n;r.tokens[r.emitted++]=greedy?best(last,n):draw(last,n,uniform(count,DS41_SPARK_TARGET));}
    r.accepted_input_rows=r.accepted_drafts+1;*out=r;return 0;
}catch(const std::exception&e){spark_error=e.what();return -1;}}

extern "C" int ds41_spark_verify(const int32_t*d,const double*q,const double*p,int count,int n,int greedy,uint64_t seed,uint64_t pos,ds41_spark_result*out){return verify_impl(d,q,p,count,n,greedy,seed,pos,out,false);}
extern "C" int ds41_spark_verify_v2(const int32_t*d,const double*q,const double*p,int count,int n,int greedy,uint64_t seed,uint64_t step,ds41_spark_result*out){return verify_impl(d,q,p,count,n,greedy,seed,step,out,true);}
extern "C" uint32_t ds41_spark_rng_word_v2(uint64_t seed,uint64_t step,uint32_t slot,uint32_t purpose,uint64_t draw){if(slot>5||purpose>3||draw/4>UINT32_MAX){spark_error="Philox v2 address out of range";return 0;}return spark_word_v2(seed,step,slot,purpose,draw);}
extern "C" double ds41_spark_uniform_v2(uint64_t seed,uint64_t step,uint32_t slot,uint32_t purpose,uint64_t draw){if(slot>5||purpose>3||draw>8589934591ULL){spark_error="Philox v2 address out of range";return NAN;}return spark_uniform_v2(seed,step,slot,purpose,draw);}
extern "C" int ds41_spark_sample_v2(const double*p,int n,uint64_t seed,uint64_t step,uint32_t slot,uint32_t purpose,int32_t*out){try{if(!out||slot>5||purpose>3)throw std::invalid_argument("sample v2 arguments");validate(p,n);*out=draw(p,n,spark_uniform_v2(seed,step,slot,purpose));return 0;}catch(const std::exception&e){spark_error=e.what();return -1;}}
