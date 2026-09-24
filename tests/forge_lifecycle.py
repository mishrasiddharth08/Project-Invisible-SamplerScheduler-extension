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
import runpy
from modules import script_callbacks
engine = p.parent / "scripts/engine.py"
for attempt in range(2):
    count=len(script_callbacks.callback_map['callbacks_script_unloaded'])
    runpy.run_path(str(engine))
    assert getattr(lib, '_pi_ss_startup_done', False)
    assert len(register.owned()['samplers'])==46
    assert len(register.owned()['schedulers'])==8
    runpy.run_path(str(engine))
    assert len(script_callbacks.callback_map['callbacks_script_unloaded'])==count+1
    script_callbacks.callback_map['callbacks_ui_settings'][-1].callback()
    assert 'pi_ss_custom_sigmas' in shared.opts.data_labels
    script_callbacks.callback_map['callbacks_script_unloaded'][-1].callback()
    assert sd_samplers.all_samplers==stock
    assert sd_schedulers.all_schedulers==stock_s
    assert not getattr(lib, '_pi_ss_startup_done', False)
# A replacement installed later under our label must survive unload.
register.register_samplers()
original=next(x for x in sd_samplers.all_samplers if x.name=='iPNDM')
replacement=type(original)(original.name,original.constructor,original.aliases,original.options)
sd_samplers.all_samplers[sd_samplers.all_samplers.index(original)]=replacement
register.unregister_all()
assert any(x is replacement for x in sd_samplers.all_samplers)
sd_samplers.all_samplers.remove(replacement)
sd_samplers.all_samplers_map={x.name:x for x in sd_samplers.all_samplers}
shared.opts.hide_schedulers=['Cosine']
register.register_schedulers()
assert any(x.name=='cosine' for x in sd_schedulers.all_schedulers)
assert not any(x.name=='cosine' for x in sd_schedulers.schedulers)
assert not register.register_schedulers()
register.unregister_all()
shared.opts.hide_schedulers=[]
register.register_samplers(['ipndm'])
assert not any(x.name=='iPNDM' for x in sd_samplers.all_samplers)
register.unregister_all()
assert sd_samplers.all_samplers==stock
assert sd_schedulers.all_schedulers==stock_s
print('PASS: engine startup, re-entry, settings, unload/reload, replacement ownership, hidden schedulers, alias denylist')
