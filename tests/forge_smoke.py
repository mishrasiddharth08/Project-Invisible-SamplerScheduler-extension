import sys,os,pathlib,importlib.util,inspect,json,traceback
from types import SimpleNamespace as NS
root=pathlib.Path(os.environ['PI_FORGE_ROOT'])
sys.path[:0]=[str(root),str(root/'modules_forge/packages')]
import torch
from modules import shared,options,shared_options
shared.options_templates=shared_options.options_templates
shared.opts=options.Options(shared_options.options_templates,shared_options.restricted_opts)
from modules import processing,sd_samplers,sd_schedulers
p=pathlib.Path(__file__).resolve().parents[1]/'lib'
spec=importlib.util.spec_from_file_location('pi_samplerscheduler_lib',p/'__init__.py',submodule_search_locations=[str(p)])
lib=importlib.util.module_from_spec(spec);sys.modules[spec.name]=lib;spec.loader.exec_module(lib)
from pi_samplerscheduler_lib import register,comfy_sync
from k_diffusion import sampling
stock=list(sd_samplers.all_samplers); stock_s=list(sd_schedulers.all_schedulers)
added=register.register_samplers(); added_s=register.register_schedulers()
print('REGISTRY',len(added),len(added_s),register.report())
assert not register.report()['refused'],register.report()
assert not register.register_samplers()
assert not register.register_schedulers()
assert len({s.name for s in sd_samplers.all_samplers}) == len(sd_samplers.all_samplers)
class Model:
 def __init__(self,pred,device):
  predictor=NS(prediction_type=pred,sigma_min=torch.tensor(0.02),sigma_max=torch.tensor(0.95 if pred=='const' else 3.),percent_to_sigma=lambda t: torch.tensor(1-t),sigma=lambda t:t,timestep=lambda t:t,noise_scale=1.0)
  self.inner_model=NS(predictor=predictor,inner_model=NS(device=device,forge_objects=NS(unet=NS(model_options={},model=NS(predictor=predictor)))))
  self.init_latent=self.mask=self.nmask=None
 def __call__(self,x,sigma,**kwargs):
  sig=sigma.reshape(sigma.shape+(1,)*(x.ndim-sigma.ndim))
  den=x/(1+sig)*0.6; unc=den*0.8
  for hook in kwargs.get('model_options',{}).get('sampler_post_cfg_function',[]):
   den=hook(dict(denoised=den,uncond_denoised=unc,cond_denoised=den,uncond=True,cond=True,input=x,sigma=sigma))
  return den
results=[]
for device in ['cpu','cuda']:
 for pred in ['epsilon','const']:
  for c in register.sampler_candidates():
   if c.label not in added: continue
   try:
    torch.manual_seed(123); m=Model(pred,torch.device(device));x=torch.randn(1,4,8,8,device=device); sigmas=torch.tensor([0.95,0.7,0.4,0.15,0.02,0.],device=device) if pred=='const' else torch.tensor([3.,2.,1.,0.4,0.02,0.],device=device)
    f=c.loader();f=getattr(sampling,f) if isinstance(f,str) else f
    kw=dict(extra_args={'seed':123},callback=lambda d:None,disable=True)
    if 'sigmas' in inspect.signature(f).parameters: kw['sigmas']=sigmas
    else:
     kw.update(sigma_min=sigmas[-2],sigma_max=sigmas[0])
     if 'n' in inspect.signature(f).parameters: kw['n']=5
    if c.options.get('solver_type'):kw['solver_type']=c.options['solver_type']
    y=f(m,x.clone(),**kw)
    assert y.shape==x.shape and y.dtype==x.dtype and y.device==x.device,(y.shape,y.dtype,y.device)
    assert torch.isfinite(y).all(),'non-finite output'
    results.append(dict(name=c.label,device=device,prediction=pred,status='pass'))
   except Exception as e:
    results.append(dict(name=c.label,device=device,prediction=pred,status='FAIL',error=str(e)))
    print('FAIL',c.label,device,pred,e);traceback.print_exc(limit=3)
from pi_samplerscheduler_lib.wrappers import schedulers
for name,label,fn,inner,_,_ in schedulers.SCHEDULER_TABLE:
 for n in [1,2,5,20,100]:
  for lo,hi in [(0.02,3.),(0.002,0.95)]:
   kw=dict(n=n,sigma_min=lo,sigma_max=hi,device='cpu')
   if inner:kw['inner_model']=NS(sigmas=torch.linspace(lo,hi,1000))
   try:
    y=fn(**kw);assert len(y)==n+1 and y[-1]==0;assert torch.isfinite(y).all();assert (y[:-1]>0).all();assert (y[:-1]>=y[1:]).all(),y
    results.append(dict(name=label,n=n,range=[lo,hi],status='pass'))
   except Exception as e:
    results.append(dict(name=label,n=n,status='FAIL',error=str(e)));print('SCHED FAIL',label,n,e)
# Upstream name coverage and scheduler equivalents.
remote_path = os.environ.get("PI_COMFY_SOURCE")
if remote_path:
    remote = comfy_sync.parse_name_lists(pathlib.Path(remote_path).read_text())
    coverage = comfy_sync.classify(remote)
    assert not coverage["needs_port"] and not coverage["schedulers_needing_port"], coverage
    pathlib.Path(os.environ["PI_COVERAGE_OUTPUT"]).write_text(json.dumps({"remote":remote,"coverage":coverage},indent=2))
# Mask rescaling reaches the real denoiser and restores its state, even on error.
from pi_samplerscheduler_lib.wrappers import samplers_cfgpp
for fail in [False, True]:
    class MaskModel(Model):
        def __call__(self, x, sigma, **kwargs):
            assert self.init_latent.shape[-2:] == x.shape[-2:]
            if fail and x.shape[-1] != 8:
                raise RuntimeError("expected test interruption")
            return super().__call__(x, sigma, **kwargs)
    m=MaskModel('epsilon',torch.device('cpu'))
    original=torch.zeros(1,4,8,8);m.init_latent=original
    try:
        samplers_cfgpp.sample_euler_dy_cfgpp(m,torch.randn_like(original),torch.tensor([3.,2.,1.,.4,.02,0.]),disable=True)
        assert not fail
    except RuntimeError as e:
        assert fail and str(e)=="expected test interruption"
    assert m.init_latent is original
    assert m.inner_model.inner_model.forge_objects.unet.model_options == {}
# Same-seed reproducibility for the new stochastic port.
from pi_samplerscheduler_lib.wrappers import comfy_latest
x=torch.randn(1,4,8,8); sigmas=torch.tensor([3.,2.,1.,.4,.02,0.])
a=comfy_latest.sample_dpmpp_2s_ancestral_cfg_pp(Model('epsilon','cpu'),x.clone(),sigmas,extra_args={'seed':17},disable=True)
b=comfy_latest.sample_dpmpp_2s_ancestral_cfg_pp(Model('epsilon','cpu'),x.clone(),sigmas,extra_args={'seed':17},disable=True)
assert torch.equal(a,b)
for expression in ["float('nan')", "M*x", "(1).__class__", "[nan, 1, 0]", "[1, 4, 2]"]:
    old=schedulers._custom_expression;schedulers._custom_expression=lambda:expression
    y=schedulers.custom_scheduler(5,.02,3.,'cpu')
    assert torch.isfinite(y).all() and (y[:-1]>=y[1:]).all()
    schedulers._custom_expression=old
register.unregister_all()
assert sd_samplers.all_samplers==stock
assert sd_schedulers.all_schedulers==stock_s
assert not register.owned()['samplers']
assert len(register.register_samplers())==len(added)
register.unregister_all()
output=os.environ.get('PI_TEST_OUTPUT','validation.json');pathlib.Path(output).write_text(json.dumps({'added_samplers':added,'added_schedulers':added_s,'cases':results},indent=2))
failed=[r for r in results if r['status']=='FAIL'];print('RESULT',len(results),'cases',len(failed),'failures');sys.exit(bool(failed))
