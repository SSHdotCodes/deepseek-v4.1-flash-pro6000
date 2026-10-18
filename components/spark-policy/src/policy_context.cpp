#include "ds41/spark_policy.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
namespace {
bool finite_nonnegative(double x){return std::isfinite(x)&&x>=0;}
bool config_ok(const ds41_spark_policy_config_v2*c){
 if(!c||c->size!=sizeof(*c)||c->version!=2||c->max_width<1||c->max_width>6||
    !(c->qualified_mask&1)||(c->qualified_mask&~63)||c->exploration_period<0||
    c->exploration_width<1||c->exploration_width>6||c->min_cost_samples<1||c->min_kind_samples<0||c->min_confidence_samples<0||
    !finite_nonnegative(c->prior_strength)||!finite_nonnegative(c->cell_strength)||
    c->prior_strength==0||c->cell_strength==0||!finite_nonnegative(c->margin)||!finite_nonnegative(c->min_preflight_gain)||
    !std::isfinite(c->confidence_cut[0])||!std::isfinite(c->confidence_cut[1])||
    c->confidence_cut[0]>=c->confidence_cut[1])return false;
 for(int i=0;i<6;++i)if(!finite_nonnegative(c->verify_ms[i])||c->verify_ms[i]==0||
   !finite_nonnegative(c->draft_ms[i])||!finite_nonnegative(c->other_ms[i]))return false;
 for(int m=0;m<4;++m){double prev=1;for(double p:c->prior_survival[m]){if(!std::isfinite(p)||p<0||p>prev)return false;prev=p;}}
 return true;
}
bool state_ok(const ds41_spark_policy_state_v2*s){return s&&s->size==sizeof(*s)&&s->version==2;}
bool features_ok(const ds41_spark_policy_features_v2*f){
 return f&&f->size==sizeof(*f)&&f->version==2&&f->known_anchor>=0&&f->known_anchor<129280&&
 f->remaining_tokens>=0&&f->max_width>=1&&f->max_width<=6&&(f->qualified_mask&1)&&!(f->qualified_mask&~63)&&
 f->mode>=0&&f->mode<4&&f->token_kind>=0&&f->token_kind<4&&
 (f->confidence_available==0||f->confidence_available==1)&&
 (!f->confidence_available||std::isfinite(f->confidence_raw))&&finite_nonnegative(f->temperature)&&
 std::isfinite(f->top_p)&&f->top_p>0&&f->top_p<=1&&finite_nonnegative(f->paid_draft_ms)&&
 (!f->confidence_available||f->paid_draft_ms>0);
}
int cell_index(const ds41_spark_policy_config_v2*c,const ds41_spark_policy_features_v2*f){
 int bin=!f->confidence_available?0:1+(f->confidence_raw>=c->confidence_cut[0])+(f->confidence_raw>=c->confidence_cut[1]);
 int profile=f->temperature==0?0:(f->temperature==1&&f->top_p==1?1:2);
 return ((f->mode*4+f->token_kind)*4+bin)*3+profile;
}
bool counts_ok(int width,int accepted,int tested){return width>=1&&width<=6&&accepted>=0&&accepted<width&&
 tested>=accepted&&tested<width&&tested<=accepted+1;}
void count(ds41_spark_policy_counts_v2&out,int accepted,int tested){
 for(int k=0;k<tested;++k){++out.reached[k];out.accepted[k]+=k<accepted;}
}
bool fresh(const ds41_spark_policy_state_v2*s,uint64_t nonce){return !s->has_nonce||nonce>s->last_nonce;}
void retire(ds41_spark_policy_state_v2*s,uint64_t nonce){s->last_nonce=nonce;s->has_nonce=1;++s->generation;}
}
extern "C" void ds41_spark_policy_default_v2(ds41_spark_policy_config_v2*c){
 if(!c)return;
 *c={};c->size=sizeof(*c);c->version=2;c->max_width=6;c->qualified_mask=63;
 c->exploration_period=16;c->exploration_width=3;c->min_cost_samples=8;
 c->min_kind_samples=16;c->min_confidence_samples=32;
 c->prior_strength=4;c->cell_strength=16;c->margin=.02;c->min_preflight_gain=.10;c->confidence_cut[0]=0;c->confidence_cut[1]=std::log(9.);
 // True verified input rows, INCLUDING anchor. Intermediate costs interpolate
 // the measured replay curve; these are model priors, not workload timings.
 double verify[]={50.,79.7,102.85,126.,146.5,167.};
 double draft[]={0.,3.766354,4.261100,4.856223,5.273624,5.834640};
 double prior[4][5]={{.5,.2,.08,.03,.01},{1./24,1./24,0,0,0},
  {13./24,3./24,2./24,0,0},{20./24,16./24,7./24,6./24,3./24}};
 std::memcpy(c->verify_ms,verify,sizeof(verify));std::memcpy(c->draft_ms,draft,sizeof(draft));
 std::memcpy(c->prior_survival,prior,sizeof(prior));
}
extern "C" int ds41_spark_policy_reset_v2(ds41_spark_policy_state_v2*s,const ds41_spark_policy_config_v2*c){
 if(!s||!config_ok(c))return -1;
 *s={};s->size=sizeof(*s);s->version=2;return 0;
}
extern "C" int ds41_spark_policy_seed_v2(ds41_spark_policy_state_v2*s,const ds41_spark_policy_config_v2*c,
 const ds41_spark_policy_features_v2*f,int width,int accepted,int tested){
 if(!state_ok(s)||!config_ok(c)||!features_ok(f)||s->generation||s->has_nonce||!counts_ok(width,accepted,tested))return -1;
 int cell=cell_index(c,f);count(s->mode[f->mode*3+cell%3],accepted,tested);
 count(s->kind[(f->mode*4+f->token_kind)*3+cell%3],accepted,tested);count(s->cell[cell],accepted,tested);return 0;
}
extern "C" int ds41_spark_policy_decide_v2(const ds41_spark_policy_state_v2*s,const ds41_spark_policy_config_v2*c,
 const ds41_spark_policy_features_v2*f,ds41_spark_policy_decision_v2*out){
 if(!out||!state_ok(s)||!config_ok(c)||!features_ok(f)||!fresh(s,f->nonce))return -1;
 *out={};out->size=sizeof(*out);out->version=2;out->nonce=f->nonce;out->generation=s->generation;
 out->mode=f->mode;out->cell=cell_index(c,f);out->paid_draft_ms=f->paid_draft_ms;if(!f->remaining_tokens)return 0;
 int cap=std::min({c->max_width,f->max_width,f->remaining_tokens});int mask=c->qualified_mask&f->qualified_mask;
 double survival=1,emitted=1,best_rate=0;int best=1;const auto&mode=s->mode[f->mode*3+out->cell%3];
 const auto&kind=s->kind[(f->mode*4+f->token_kind)*3+out->cell%3];const auto&cell=s->cell[out->cell];
 for(int w=1;w<=cap;++w){int i=w-1;if(i){int k=i-1;double prev=k?c->prior_survival[f->mode][k-1]:1;
   double prior=prev>0?c->prior_survival[f->mode][k]/prev:.5;
   // Off-profile sampling uses a conservative conditional prior. Online cells
   // remain separated by temperature profile, kind and pre-sample confidence.
   if(f->temperature!=1||f->top_p!=1)prior=.5;
   double coarse=(mode.accepted[k]+c->prior_strength*prior)/(mode.reached[k]+c->prior_strength);
   if(kind.reached[0]>=uint64_t(c->min_kind_samples))coarse=(kind.accepted[k]+c->cell_strength*coarse)/(kind.reached[k]+c->cell_strength);
   double fine=coarse;
   if(f->confidence_available&&cell.reached[0]>=uint64_t(c->min_confidence_samples))fine=(cell.accepted[k]+c->cell_strength*coarse)/(cell.reached[k]+c->cell_strength);
   survival*=std::clamp(fine,0.,1.);emitted+=survival;
  }
  double blend=std::min(1.,double(s->cost_samples[i])/c->min_cost_samples);
  double verify=c->verify_ms[i]+blend*(s->verify_ms[i]-c->verify_ms[i]);
  double draft=c->draft_ms[i]+blend*(s->draft_ms[i]-c->draft_ms[i]);
  double other=c->other_ms[i]+blend*(s->other_ms[i]-c->other_ms[i]);
  double cost=verify+std::max(draft,f->paid_draft_ms)+other;
  out->predicted_emitted[i]=emitted;out->predicted_ms[i]=cost;out->predicted_tps[i]=1000*emitted/cost;
  out->predicted_host_misses[i]=s->host_misses[i];out->predicted_nvme_misses[i]=s->nvme_misses[i];
  if((mask&(1<<i))&&out->predicted_tps[i]>best_rate*(1+c->margin)){best=w;best_rate=out->predicted_tps[i];}
 }
 // Require a material modeled gain before paying the cold drafter cost.
 // Once forward/q0 is paid, compare remaining choices with that sunk cost in
 // every candidate instead; do not erase it or apply the startup threshold.
 if(f->paid_draft_ms==0&&best_rate<out->predicted_tps[0]*(1+c->min_preflight_gain))best=1;
 if(c->exploration_period>0&&(f->nonce+uint64_t(1))%uint64_t(c->exploration_period)==0){
  int w=std::min(c->exploration_width,cap);while(w>1&&!(mask&(1<<(w-1))))--w;best=w;out->exploration=1;
 }
 out->width=best;return 0;
}
extern "C" int ds41_spark_policy_observe_v2(ds41_spark_policy_state_v2*s,const ds41_spark_policy_config_v2*c,
 const ds41_spark_policy_decision_v2*d,const ds41_spark_policy_feedback_v2*f){
 if(!state_ok(s)||!config_ok(c)||!d||d->size!=sizeof(*d)||d->version!=2||d->generation!=s->generation||
 !fresh(s,d->nonce)||d->mode<0||d->mode>3||d->cell<0||d->cell>=DS41_SPARK_POLICY_CELLS||d->cell/48!=d->mode||
 !f||f->size!=sizeof(*f)||f->version!=2||f->width!=d->width||
 !counts_ok(f->width,f->accepted_drafts,f->tested_drafts)||(f->timing_valid!=0&&f->timing_valid!=1)||
 !finite_nonnegative(f->verify_ms)||!finite_nonnegative(f->draft_ms)||!finite_nonnegative(f->other_ms)||
 !finite_nonnegative(f->host_misses)||!finite_nonnegative(f->nvme_misses)||f->nvme_misses>f->host_misses||
 (f->timing_valid&&(f->verify_ms==0||f->draft_ms+1e-6<d->paid_draft_ms)))return -1;
 count(s->mode[d->mode*3+d->cell%3],f->accepted_drafts,f->tested_drafts);
 count(s->kind[(d->cell/12)*3+d->cell%3],f->accepted_drafts,f->tested_drafts);
 count(s->cell[d->cell],f->accepted_drafts,f->tested_drafts);
 if(f->timing_valid){int i=f->width-1;double a=1./std::min<uint64_t>(++s->cost_samples[i],32);
  s->verify_ms[i]+=a*(f->verify_ms-s->verify_ms[i]);s->draft_ms[i]+=a*(f->draft_ms-s->draft_ms[i]);
  s->other_ms[i]+=a*(f->other_ms-s->other_ms[i]);s->host_misses[i]+=a*(f->host_misses-s->host_misses[i]);s->nvme_misses[i]+=a*(f->nvme_misses-s->nvme_misses[i]);
 }
 retire(s,d->nonce);++s->completed;return 0;
}
extern "C" int ds41_spark_policy_abort_v2(ds41_spark_policy_state_v2*s,uint64_t nonce){
 if(!state_ok(s)||!fresh(s,nonce))return -1;
 retire(s,nonce);return 0;
}
