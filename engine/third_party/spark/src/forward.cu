#include "ds41/spark.h"
#include "append_tail.hpp"
#include "ds41/dense.h"
#include "ds41/attention.h"
#include "ds41/state.hpp"
#include "ds41/kernels.cuh"
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <map>
#include <string>
#include <stdexcept>
#include <utility>
#include <algorithm>
#include <limits>
extern thread_local std::string spark_error;
namespace {
using bf16=__nv_bfloat16;
constexpr int H=5120,I=2304,V=129280,B=5;
void ck(int e,const char*msg){if(e)throw std::runtime_error(std::string(msg)+" status="+std::to_string(e));}
#define CK(x) ck(int(x),#x)
struct Buf{
 void*p=nullptr;size_t n=0;Buf()=default;Buf(size_t bytes):n(bytes){CK(cudaMalloc(&p,n));}
 ~Buf(){if(p)cudaFree(p);}Buf(const Buf&)=delete;Buf&operator=(const Buf&)=delete;
 template<class T=void>T*at(size_t off=0)const{return reinterpret_cast<T*>((char*)p+off);}
};
__global__ void embed(const bf16*w,bf16*h,float*pre,int anchor){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<B*4*H){int row=i/(4*H),col=i%H;h[i]=w[size_t(row?128799:anchor)*H+col];}if(i<B*4)pre[i]=(i%4==0)?1.f:0.f;}
__device__ void top3_swap(float*v,int*ids,bool*valid,int a,int b,bool dir){bool swap=(v[a]>v[b]&&valid[a])||!valid[b];if(swap==dir){float f=v[a];v[a]=v[b];v[b]=f;int id=ids[a];ids[a]=ids[b];ids[b]=id;bool q=valid[a];valid[a]=valid[b];valid[b]=q;}}
__global__ void select3(const float*s,int32_t*ids,int M){int row=blockIdx.x;if(threadIdx.x||row>=M)return;
 int chosen[3]={-1,-1,-1};for(int k=0;k<3;++k){int best=-1;for(int i=0;i<128;++i){bool used=false;for(int j=0;j<k;++j)used|=chosen[j]==i;if(!used&&(best<0||s[row*128+i]>s[row*128+best]))best=i;}chosen[k]=best;}
 // CUDA torch.topk first gathers strict winners in index order, then the
 // threshold ties in index order. The pinned Torch CUDA sorted path uses
 // the4-entry network for one row and32-entry padded network for multiple
 // rows; the latter reverses a tied pair above the threshold as well.
 float threshold=s[row*128+chosen[2]],v[32]={};int order[32]={};bool valid[32]={true,true,true};int count=0;
 for(int i=0;i<128;++i)if(s[row*128+i]>threshold){v[count]=s[row*128+i];order[count++]=i;}
 for(int i=0;i<128&&count<3;++i)if(s[row*128+i]==threshold){v[count]=s[row*128+i];order[count++]=i;}
 int size_limit=M==1?4:32;
 for(int size=2;size<size_limit;size*=2)for(int stride=size/2;stride;stride/=2)for(int t=0;t<size_limit/2;++t){int a=2*t-(t&(stride-1));top3_swap(v,order,valid,a,a+stride,bool(t&(size/2)));}
 for(int stride=size_limit/2;stride;stride/=2)for(int t=0;t<size_limit/2;++t){int a=2*t-(t&(stride-1));top3_swap(v,order,valid,a,a+stride,false);}
 for(int k=0;k<3;++k)ids[row*3+k]=order[k];}
/* Ring physical slots remain in official slot order even after wrap. */
__global__ void gather_block(const bf16*ring,const bf16*block,bf16*out,int32_t*idx,int count){int i=blockIdx.x*blockDim.x+threadIdx.x;int width=count+B;
 if(i<B*width*512){int col=(i/512)%width,k=i%512;out[i]=col<count?ring[col*512+k]:block[(col-count)*512+k];}
 if(i<B*width){int col=i%width;idx[i]=col;}}
__global__ void markov_input(const bf16*hd,const bf16*emb,float*out){int i=threadIdx.x+blockIdx.x*blockDim.x;if(i<B*(H+256)){int row=i/(H+256),col=i%(H+256);out[i]=__bfloat162float(col<H?hd[row*H+col]:emb[col-H]);}}
__global__ void add_bias(const float*u,const float*b,float*out){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<V)out[i]=__fadd_rn(u[i],b[i]);}
__global__ void gather_expert(const bf16*x,const int32_t*ids,const float*w,bf16*out,float*rw,int expert,int count){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count*H)return;int selected=i/H,k=i%H,seen=0;for(int s=0;s<B*3;++s)if(ids[s]==expert){if(seen++==selected){out[i]=x[(s/3)*H+k];if(!k)rw[selected]=w[s];break;}}}
__global__ void scatter_expert(const bf16*input,const int32_t*ids,bf16*out,int expert,int count){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count*H)return;int selected=i/H,k=i%H,seen=0;for(int s=0;s<B*3;++s)if(ids[s]==expert){if(seen++==selected){out[s*H+k]=input[i];break;}}}
/* sol-dense Reduce.cuh geometry extended to cold seed rows. The launch shape
 * depends on the original full prompt, even when only its last128 rows remain. */
__device__ float sum_seed(float v,float*sh,int width){int t=threadIdx.x;sh[t]=v;__syncthreads();for(int d=width/2;d>=32;d>>=1){if(t%width<d){v=__fadd_rn(v,sh[t+d]);sh[t]=v;}__syncthreads();}for(int d=16;d;d>>=1)v=__fadd_rn(v,__shfl_down_sync(0xffffffff,v,d));if(t%width==0)sh[t]=v;__syncthreads();for(int d=(blockDim.x/width)/2;d;d>>=1){if(t%width==0&&t/width<d){v=__fadd_rn(v,sh[t+d*width]);sh[t]=v;}__syncthreads();}return sh[0];}
__global__ void rms_seed(const bf16*x,const bf16*w,bf16*y,int K,int width){__shared__ float sh[512];int t=threadIdx.x,row=blockIdx.x;float v[4]={0,0,0,0};for(int i=t*4;i<K;i+=blockDim.x*4)for(int j=0;j<4;++j)if(i+j<K){float a=__bfloat162float(x[row*K+i+j]);v[j]=__fadd_rn(v[j],__fmul_rn(a,a));}float z=sum_seed(__fadd_rn(__fadd_rn(__fadd_rn(v[0],v[1]),v[2]),v[3]),sh,width);if(!t)sh[0]=rsqrtf(__fadd_rn(__fmul_rn(z,1.f/K),1e-20f));__syncthreads();for(int i=t;i<K;i+=blockDim.x)y[row*K+i]=__float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(x[row*K+i]),sh[0]),__bfloat162float(w[i])));}
}
struct ds41_spark{
 ds41_spark_config c;cudaStream_t stream;cublasHandle_t blas;int64_t committed=0;bool drafted=false;bool poisoned=false;bool capture=false;
 std::map<std::string,ds41_spark_tensor>w;
 Buf h{B*4*H*2},h2{B*4*H*2},x{8*H*2},main{128*H*2},y{B*H*2},pre{B*4*4},ap{B*4*4},fp{B*4*4},post{B*4*4},comb{B*16*4},mix{B*24*4},rstd{B*4},promoted{B*4*H*4};
 Buf qa{B*1280*2},q{B*32768*2},kv{128*512*2},o{B*32768*2},wo{B*8192*2},
     gather{B*133*512*2},idx{B*133*4},aq{8*32768},as{8*1024},
     router{B*128*4},unbiased{B*128*4},biased{B*128*4},ids{B*3*4},route{B*3*4},
     routed{B*3*H*2},sg{B*I*2},su{B*I*2},sa{B*I*2},shared{B*H*2},
     U{B*V*4},hd{B*H*2},biased_logits{V*4},markov_bias{V*4},markov_x{256*4},confidence_x{B*(H+256)*4},confidence{B*4},packed{8*528};
 Buf rings[3]{{128*512*2},{128*512*2},{128*512*2}};
 Buf eq{8*H},es{8*(H/32)},eg{8*I*2},eu{8*I*2},ez{8*I*2},ezq{8*I},ezs{8*I/32};
 Buf ex{B*H*2},er{B*4},eo{B*H*2};
 explicit ds41_spark(const ds41_spark_config&v):c(v),stream((cudaStream_t)v.stream),blas((cublasHandle_t)v.blas_handle){}
 const ds41_spark_tensor& weight(const std::string&n){auto it=w.find(n);if(it==w.end())throw std::runtime_error("missing binding "+n);return it->second;}
 const void* ptr(const std::string&n){return weight(n).data;}
 void debug(const std::string&name,const void*data,size_t bytes){if(!capture)return;auto it=w.find("debug."+name);if(it!=w.end()){auto&t=it->second;size_t element=t.dtype==DS41_SPARK_F32?4:t.dtype==DS41_SPARK_BF16?2:1;if(bytes>size_t(t.rows)*t.cols*element)throw std::invalid_argument("debug buffer too small "+name);CK(cudaMemcpyAsync((void*)t.data,data,bytes,cudaMemcpyDeviceToDevice,stream));}}
 void linear(const std::string&p,const void*in,void*out,int m,int n,int k){const auto&t=weight(p+".weight");if(t.rows!=n||t.cols!=k)throw std::runtime_error("shape "+p);debug(p+".input",in,size_t(m)*k*(t.dtype==DS41_SPARK_F32?4:2));
  if(t.dtype==DS41_SPARK_FP8){CK(ds41_dense_quant(in,aq.p,as.p,m,k,stream));CK(ds41_dense_fp8(aq.p,as.p,t.data,ptr(p+".scale"),out,m,n,k,-1,stream));}
  else CK(ds41_dense_cublas(blas,in,t.data,out,m,n,k,t.dtype==DS41_SPARK_F32,1,stream));debug(p+".output",out,size_t(m)*n*(t.dtype==DS41_SPARK_F32?4:2));
 }
 void norm(const std::string&p,const void*in,void*out,int m,int k){debug(p+".input",in,size_t(m)*k*2);CK(ds41_dense_rms(in,ptr(p+".weight"),out,m,k,1e-20f,stream));debug(p+".output",out,size_t(m)*k*2);}
 void rope(void*data,int m,int heads,int dim,int64_t pos,bool inverse=false){const auto&f=weight("rope");if(pos<0||pos+m>f.rows||f.cols!=64||f.dtype!=DS41_SPARK_F32)throw std::runtime_error("RoPE position/shape");CK(ds41_dense_rope(data,(const float*)f.data+pos*64,m,heads,dim,64,inverse,stream));}
 void project(const void*captures,void*out,int rows){linear("mtp.0.main_proj",captures,out,rows,H,H*3);norm("mtp.0.main_norm",out,out,rows,H);}
 void seed_norm(const std::string&p,void*data,int count,int K,int64_t total){int height=1;while(height*2<=total&&height*2<=16)height*=2;int n=1;while(n*2<=K/4&&n*2<=512)n*=2;int width=std::min(n,512/height);int threads=K/width>=height*16?width*height:width;rms_seed<<<count,threads,0,stream>>>((const bf16*)data,(const bf16*)ptr(p+".weight"),(bf16*)data,K,width);CK(cudaGetLastError());}
 void seed(const void*captures,int64_t total){if(total<1)throw std::invalid_argument("seed positive prompt length");int count=int(std::min<int64_t>(128,total));int64_t start=total-count;
  for(int i=0;i<count;i+=8)linear("mtp.0.main_proj",(const bf16*)captures+size_t(i)*3*H,main.at<bf16>(size_t(i)*H*2),std::min(8,count-i),H,3*H);
  seed_norm("mtp.0.main_norm",main.p,count,H,total);
  for(int l=0;l<3;++l){CK(cudaMemsetAsync(rings[l].p,0,rings[l].n,stream));std::string p="mtp."+std::to_string(l)+".attn";for(int i=0;i<count;i+=8)linear(p+".wkv",main.at<bf16>(size_t(i)*H*2),kv.at<bf16>(size_t(i)*512*2),std::min(8,count-i),512,H);
   seed_norm(p+".kv_norm",kv.p,count,512,total);rope(kv.p,count,1,512,start);CK(ds41_attn_roundtrip(kv.p,count,512,DS41_SWA_FP8,stream));CK(ds41_attn_ring_write(kv.p,rings[l].p,512*2,start,count,128,stream));}
  committed=total;drafted=false;poisoned=false;
 }
 void ingest(const void*mx,int64_t start,int rows){
  if(poisoned)throw std::runtime_error("state poisoned; reset required");
  if(start!=committed||rows<1||rows>8)throw std::invalid_argument("ingest contiguous committed rows1..8");
  for(int l=0;l<3;++l){std::string p="mtp."+std::to_string(l)+".attn";linear(p+".wkv",mx,kv.p,rows,512,H);norm(p+".kv_norm",kv.p,kv.p,rows,512);rope(kv.p,rows,1,512,start);CK(ds41_attn_roundtrip(kv.p,rows,512,DS41_SWA_FP8,stream));CK(ds41_attn_ring_write(kv.p,rings[l].p,512*2,start,rows,128,stream));}
  committed+=rows;drafted=false;
 }
 // Ported from frozen append-tail-m1-v1: every supplied tail row uses the unchanged M1 projection and
 // ingest write path; the private frontier jump publishes once after the whole tail is enqueued.
 void restore_append_tail_m1(const ds41_spark_append_tail_v1*d){
  const auto plan=ds41_spark_detail::checked_append_tail(d,committed,c.stream,std::numeric_limits<int64_t>::max(),capture,poisoned);
  const auto&f=weight("rope");if(f.dtype!=DS41_SPARK_F32||f.cols!=64||f.rows<plan.end)throw std::invalid_argument("append restore: RoPE binding/extent");
  cudaStreamCaptureStatus status;CK(cudaStreamIsCapturing(stream,&status));if(status!=cudaStreamCaptureStatusNone)throw std::invalid_argument("append restore: active stream capture");
  cudaPointerAttributes attributes{};CK(cudaPointerGetAttributes(&attributes,d->captures_bf16));int device;CK(cudaGetDevice(&device));
  if(attributes.type!=cudaMemoryTypeDevice||attributes.device!=device)throw std::invalid_argument("append restore: captures must be device memory on the bound device");
  ds41_spark_detail::apply_append_tail(plan,committed,drafted,poisoned,[&](const void*row,int64_t position){
   project(row,main.p,1);
   for(int l=0;l<3;++l){std::string p="mtp."+std::to_string(l)+".attn";linear(p+".wkv",main.p,kv.p,1,512,H);norm(p+".kv_norm",kv.p,kv.p,1,512);rope(kv.p,1,1,512,position);CK(ds41_attn_roundtrip(kv.p,1,512,DS41_SWA_FP8,stream));CK(ds41_attn_ring_write(kv.p,rings[l].p,512*2,position,1,128,stream));}
  });
 }
 void hc(const std::string&p,const std::string&kind,Buf&target){CK(ds41_dense_hc_mixes(blas,h.p,(const float*)ptr(p+".hc_"+kind+"_fn"),(const float*)ptr(p+".hc_"+kind+"_scale"),(const float*)ptr(p+".hc_"+kind+"_base"),target.at<float>(),post.at<float>(),comb.at<float>(),promoted.at<float>(),mix.at<float>(),rstd.at<float>(),B,H,1e-20f,1e-6f,stream));}
 void attention(int l){std::string p="mtp."+std::to_string(l)+".attn";
  linear(p+".wq_a",x.p,qa.p,B,1280,H);norm(p+".q_norm",qa.p,qa.p,B,1280);linear(p+".wq_b",qa.p,q.p,B,32768,1280);rope(q.p,B,64,512,committed);
  linear(p+".wkv",x.p,kv.p,B,512,H);norm(p+".kv_norm",kv.p,kv.p,B,512);rope(kv.p,B,1,512,committed);CK(ds41_attn_roundtrip(kv.p,B,512,DS41_SWA_FP8,stream));
  int count=int(std::min<int64_t>(128,committed));int width=count+B;gather_block<<<(B*width*512+255)/256,256,0,stream>>>(rings[l].at<bf16>(),kv.at<bf16>(),gather.at<bf16>(),idx.at<int32_t>(),count);CK(cudaGetLastError());
  CK(ds41_attn_sparse(q.p,gather.p,(const float*)ptr(p+".attn_sink"),idx.at<int32_t>(),o.p,B,width,8,stream));rope(o.p,B,64,512,committed,true);CK(ds41_attn_roundtrip(o.p,B*8,4096,DS41_SWA_FP8,stream));
  CK(ds41_dense_wo_a(blas,o.p,ptr(p+".wo_a.weight"),wo.p,B,1024,4096,8,stream));linear(p+".wo_b",wo.p,y.p,B,H,8192);
 }
 void bound_experts(int l){int32_t host_ids[B*3];CK(cudaMemcpyAsync(host_ids,ids.p,sizeof(host_ids),cudaMemcpyDeviceToHost,stream));CK(cudaStreamSynchronize(stream));std::map<int,int>counts;for(int id:host_ids){if(id<0||id>=128)throw std::runtime_error("invalid top3 id");++counts[id];}
  for(auto entry:counts){int id=entry.first,m=entry.second;std::string p="mtp."+std::to_string(l)+".ffn.experts."+std::to_string(id);const void*views[6]={ptr(p+".w1.weight"),ptr(p+".w1.scale"),ptr(p+".w3.weight"),ptr(p+".w3.scale"),ptr(p+".w2.weight"),ptr(p+".w2.scale")};
   gather_expert<<<(m*H+255)/256,256,0,stream>>>(x.at<bf16>(),ids.at<int32_t>(),route.at<float>(),ex.at<bf16>(),er.at<float>(),id,m);CK(cudaGetLastError());CK(ds41_spark_expert_raw(this,views,ex.p,er.at<float>(),eo.p,m));scatter_expert<<<(m*H+255)/256,256,0,stream>>>(eo.at<bf16>(),ids.at<int32_t>(),routed.at<bf16>(),id,m);CK(cudaGetLastError());}
 }
 void moe(int l){std::string p="mtp."+std::to_string(l)+".ffn";CK(ds41_dense_promote(x.p,promoted.at<float>(),B*H,stream));linear(p+".gate",promoted.p,router.p,B,128,H);
  CK(ds41_dense_router_scores(router.at<float>(),(const float*)ptr(p+".gate.bias"),unbiased.at<float>(),biased.at<float>(),B,128,stream));
  if(c.topk)CK(c.topk(c.user,biased.at<float>(),ids.at<int32_t>(),B,c.stream));else{select3<<<B,32,0,stream>>>(biased.at<float>(),ids.at<int32_t>(),B);CK(cudaGetLastError());}
  CK(ds41_dense_router_weights(unbiased.at<float>(),ids.at<int32_t>(),route.at<float>(),B,128,3,1.5f,stream));debug("draft/layer"+std::to_string(l+40)+"/route_id_i32",ids.p,B*3*4);debug("draft/layer"+std::to_string(l+40)+"/route_weight_f32",route.p,B*3*4);
  if(c.experts)CK(c.experts(c.user,l,x.p,ids.at<int32_t>(),route.at<float>(),routed.p,B,c.stream));else bound_experts(l);
  linear(p+".shared_experts.w1",x.p,sg.p,B,I,H);linear(p+".shared_experts.w3",x.p,su.p,B,I,H);CK(ds41_dense_shared_act(sg.p,su.p,sa.p,aq.p,as.p,B,I,stream));
  linear(p+".shared_experts.w2",sa.p,shared.p,B,H,I);CK(ds41::reduce_experts(routed.at<bf16>(),ids.at<int32_t>(),shared.at<bf16>(),y.at<bf16>(),B,3,H,stream));
 }
 void forward(int anchor){if(poisoned||committed<1||anchor<0||anchor>=V)throw std::invalid_argument("forward state/anchor");capture=true;
  const auto&e=weight("embed.weight");if(e.dtype!=DS41_SPARK_BF16||e.rows!=V||e.cols!=H)throw std::runtime_error("embedding shape/dtype");
  embed<<<(B*4*H+255)/256,256,0,stream>>>((const bf16*)e.data,h.at<bf16>(),pre.at<float>(),anchor);CK(cudaGetLastError());
  for(int l=0;l<3;++l){std::string p="mtp."+std::to_string(l);hc(p,"attn",ap);CK(ds41_dense_hc_pre(h.p,pre.at<float>(),x.p,B,H,stream));norm(p+".attn_norm",x.p,x.p,B,H);attention(l);CK(ds41_dense_hc_post(y.p,h.p,post.at<float>(),comb.at<float>(),h2.p,B,H,stream));std::swap(h.p,h2.p);
   hc(p,"ffn",fp);CK(ds41_dense_hc_pre(h.p,ap.at<float>(),x.p,B,H,stream));norm(p+".ffn_norm",x.p,x.p,B,H);moe(l);CK(ds41_dense_hc_post(y.p,h.p,post.at<float>(),comb.at<float>(),h2.p,B,H,stream));std::swap(h.p,h2.p);std::swap(pre.p,fp.p);debug("draft/layer"+std::to_string(l+40)+"/hc_hidden_bf16",h.p,B*4*H*2);debug("draft/layer"+std::to_string(l+40)+"/next_pre_mix_f32",pre.p,B*4*4);
  }
  CK(ds41_dense_hc_pre(h.p,pre.at<float>(),hd.p,B,H,stream));norm("mtp.2.norm",hd.p,x.p,B,H);CK(ds41_dense_promote(x.p,promoted.at<float>(),B*H,stream));
  // Native BF16 target head bound as draft.head.weight (engine --fused-head): compact BF16 head, draft logits only.
  if(w.count("draft.head.weight"))CK(ds41_dense_head_compact(promoted.at<float>(),w.at("draft.head.weight").data,U.at<float>(),B,V,H,stream));
  else linear("head",promoted.p,U.p,B,V,H);drafted=true;capture=false;
 }
 void markov(int k,int previous){if(!drafted||k<0||k>=5||previous<0||previous>=V)throw std::invalid_argument("Markov state/k/token");const auto&e=weight("mtp.2.markov_head.embed.weight");if(e.dtype!=DS41_SPARK_BF16||e.rows!=V||e.cols!=256)throw std::runtime_error("Markov embedding shape/dtype");
  const auto*emb=(const bf16*)e.data+size_t(previous)*256;CK(ds41_dense_promote(emb,markov_x.at<float>(),256,stream));linear("mtp.2.markov_head.head",markov_x.p,markov_bias.p,1,V,256);
  add_bias<<<(V+255)/256,256,0,stream>>>(U.at<float>(size_t(k)*V*4),markov_bias.at<float>(),biased_logits.at<float>());CK(cudaGetLastError());
  markov_input<<<(B*(H+256)+255)/256,256,0,stream>>>(hd.at<bf16>(),emb,confidence_x.at<float>());CK(cudaGetLastError());linear("mtp.2.confidence_head.proj",confidence_x.p,confidence.p,B,1,H+256);
 }
};
#define SPARK_TRY try
#define SPARK_CATCH catch(const std::exception&e){spark_error=e.what();return -1;}
extern "C" int ds41_spark_create(const ds41_spark_config*c,ds41_spark**out){SPARK_TRY{if(!c||!out||c->abi_version!=1||!c->blas_handle)throw std::invalid_argument("config ABI/handle");*out=nullptr;*out=new ds41_spark(*c);return 0;}SPARK_CATCH}
extern "C" void ds41_spark_destroy(ds41_spark*s){if(s){cudaStreamSynchronize(s->stream);delete s;}}
extern "C" int ds41_spark_bind(ds41_spark*s,const char*n,ds41_spark_tensor t){SPARK_TRY{if(!s||!n||!t.data||t.rows<1||t.cols<1||t.dtype<0||t.dtype>3)throw std::invalid_argument("binding");std::string name=n;auto ends=[&](const std::string&suffix){return name.size()>=suffix.size()&&name.compare(name.size()-suffix.size(),suffix.size(),suffix)==0;};bool f32=name=="head.weight"||ends(".ffn.gate.weight")||ends(".markov_head.head.weight")||ends(".confidence_head.proj.weight");if(f32&&t.dtype!=DS41_SPARK_F32)throw std::invalid_argument("binding must be exactly promoted FP32: "+name);s->w[n]=t;return 0;}SPARK_CATCH}
extern "C" int ds41_spark_project(ds41_spark*s,const void*h,void*out,int rows){SPARK_TRY{if(!s||!h||!out||rows<1||rows>8)throw std::invalid_argument("project rows1..8");s->project(h,out,rows);return 0;}SPARK_CATCH}
extern "C" int ds41_spark_seed_projected(ds41_spark*s,const void*x,int64_t start,int rows){SPARK_TRY{if(!s||!x)throw std::invalid_argument("seed arguments");s->ingest(x,start,rows);return 0;}SPARK_CATCH}
extern "C" int ds41_spark_ingest(ds41_spark*s,const void*h,int64_t start,int rows){SPARK_TRY{if(!s||!h||rows<1||rows>8||start!=s->committed)throw std::invalid_argument("ingest arguments");for(int i=0;i<rows;++i){s->project((const bf16*)h+size_t(i)*3*H,s->main.p,1);s->ingest(s->main.p,start+i,1);}return 0;}SPARK_CATCH}
extern "C" int ds41_spark_seed(ds41_spark*s,const void*h,int64_t rows){SPARK_TRY{if(!s||!h)throw std::invalid_argument("seed arguments");s->seed(h,rows);return 0;}SPARK_CATCH}
extern "C" int ds41_spark_forward(ds41_spark*s,int32_t anchor,const float**u,const void**hd){SPARK_TRY{if(!s||!u||!hd)throw std::invalid_argument("forward arguments");s->forward(anchor);*u=s->U.at<float>();*hd=s->hd.p;return 0;}SPARK_CATCH}
extern "C" int ds41_spark_markov(ds41_spark*s,int k,int32_t previous,const float**logits,const float**conf){SPARK_TRY{if(!s||!logits||!conf)throw std::invalid_argument("Markov arguments");s->markov(k,previous);*logits=s->biased_logits.at<float>();*conf=s->confidence.at<float>(k*4);return 0;}SPARK_CATCH}
extern "C" int ds41_spark_rollback(ds41_spark*s,int64_t committed){SPARK_TRY{if(!s||committed!=s->committed)throw std::invalid_argument("rollback length must match ingested committed prefix");if(s->c.rollback)CK(s->c.rollback(s->c.user,committed,s->c.stream));s->drafted=false;return 0;}SPARK_CATCH}
extern "C" int ds41_spark_attn_rollback(void*raw,int64_t committed,void*){SPARK_TRY{if(!raw)throw std::invalid_argument("attention state null");auto*s=(ds41::VerifyState*)raw;if(s->pending_width){int64_t rows=committed-s->pending_start;if(rows<0||rows>s->pending_width)throw std::invalid_argument("attention committed prefix");s->finish(int(rows));}else if(s->committed!=committed)throw std::invalid_argument("attention state length mismatch");return 0;}SPARK_CATCH}
extern "C" int ds41_spark_topk3_gpu(const float*s,int32_t*ids,int rows,void*stream){if(!s||!ids||rows<1)return cudaErrorInvalidValue;select3<<<rows,32,0,(cudaStream_t)stream>>>(s,ids,rows);return cudaGetLastError();}
extern "C" int64_t ds41_spark_committed(const ds41_spark*s){return s?s->committed:-1;}
extern "C" int ds41_spark_restore_append_tail_m1(ds41_spark*s,const ds41_spark_append_tail_v1*d){SPARK_TRY{if(!s)throw std::invalid_argument("append restore: null context");s->restore_append_tail_m1(d);return 0;}SPARK_CATCH}
extern "C" int ds41_spark_debug_view(ds41_spark*s,const char*name,const void**out,size_t*bytes){SPARK_TRY{if(!s||!name||!out||!bytes)throw std::invalid_argument("debug view arguments");Buf*b=nullptr;std::string n=name;if(n=="main")b=&s->main;else if(n=="hidden")b=&s->h;else if(n=="pre")b=&s->pre;else if(n=="U")b=&s->U;else if(n.size()==5&&n.compare(0,4,"ring")==0&&n[4]>='0'&&n[4]<='2')b=&s->rings[n[4]-'0'];if(!b)throw std::invalid_argument("unknown debug view");*out=b->p;*bytes=b->n;return 0;}SPARK_CATCH}
extern "C" int ds41_spark_reset(ds41_spark*s){SPARK_TRY{if(!s)throw std::invalid_argument("reset null");for(int i=0;i<3;++i)CK(cudaMemsetAsync(s->rings[i].p,0,s->rings[i].n,s->stream));s->committed=0;s->drafted=false;s->poisoned=false;return 0;}SPARK_CATCH}
extern "C" int ds41_spark_expert_raw(ds41_spark*s,const void*const w[6],const void*x,const float*route,void*out,int m){SPARK_TRY{
 if(!s||!w||!x||!route||!out||m<1||m>8)throw std::invalid_argument("expert arguments");for(int i=0;i<6;++i)if(!w[i])throw std::invalid_argument("expert weights");
 CK(ds41_quantize_mxfp8_raw(x,s->eq.at<uint8_t>(),s->es.at<uint8_t>(),m,H,s->c.stream));
 CK(ds41_mxfp4_native_parallel_raw((const uint8_t*)w[0],(const uint8_t*)w[1],s->eq.at<uint8_t>(),s->es.at<uint8_t>(),s->eg.p,m,I,H,s->c.stream));
 CK(ds41_mxfp4_native_parallel_raw((const uint8_t*)w[2],(const uint8_t*)w[3],s->eq.at<uint8_t>(),s->es.at<uint8_t>(),s->eu.p,m,I,H,s->c.stream));
 CK(ds41_swiglu_raw(s->eg.p,s->eu.p,route,s->ez.p,m,I,10,s->c.stream));
 CK(ds41_quantize_mxfp8_raw(s->ez.p,s->ezq.at<uint8_t>(),s->ezs.at<uint8_t>(),m,I,s->c.stream));
 CK(ds41_mxfp4_native_parallel_raw((const uint8_t*)w[4],(const uint8_t*)w[5],s->ezq.at<uint8_t>(),s->ezs.at<uint8_t>(),out,m,H,I,s->c.stream));return 0;
}SPARK_CATCH}
