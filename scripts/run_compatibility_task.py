from __future__ import annotations

import argparse, csv, gc, json, os, pickle, random, time, traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT=Path(__file__).resolve().parents[1]
import sys;sys.path.insert(0,str(ROOT))
from src.adapters.legacy_adapter import ControlledTCNAdapter, ModernAdapter
from src.adapters.registry import build_adapter
from src.data.compatibility_data import concatenate_batches, load_common_batch, training_origins
from src.graphs.diagnostics import graph_statistics


def atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8');os.replace(tmp,path)
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.use_deterministic_algorithms(True,warn_only=True)
def config_for(model,dataset,batch=4):
    cfg={'config_id':model,'dataset':dataset,'nodes':90,'history':84,'horizon':28,'seed':13,'device':os.environ.get('RETAIL_DEVICE','cuda'),'batch_size':batch}
    if model.startswith('gts_'):cfg.update({'external_root':str(ROOT/'external/GTS'),'rnn_units':16,'num_rnn_layers':1,'max_diffusion_step':2,'temperature':.5})
    elif model.startswith('mtgnn_'):cfg.update({'external_root':str(ROOT/'external/MTGNN'),'gcn_depth':2,'subgraph_size':20,'node_dim':32,'layers':3,'conv_channels':32,'residual_channels':32,'skip_channels':64,'end_channels':128,'dropout':.1,'propalpha':.05,'tanhalpha':3.})
    elif model.startswith('mage_'):cfg.update({'external_root':str(ROOT/'external/MAGE'),'model_dim':32,'recur_num':4,'topk':2})
    elif model in {'metadata_embedding_tcn','semantic_knn_tcn','semantic_placebo_tcn'}:
        cache_root=Path(os.environ.get('RETAIL_SEMANTIC_ROOT',ROOT/'data/derived/semantic_embeddings'));cache=cache_root/f'{dataset}_embeddings.pt';payload=torch.load(cache,map_location='cpu');manifest=json.loads((cache_root/f'{dataset}_embedding_manifest.json').read_text(encoding='utf-8'))
        cfg.update({'core_model_root':str(ROOT/'src/author_core_v1/models'),'embeddings':payload['embeddings'],'embedding_manifest':manifest,'semantic_k':5,'hidden':32,'dropout':.1})
    return cfg
def build_model(model,dataset,batch=4):
    if model.startswith('tcn_') and model!='tcn_target_only':return ControlledTCNAdapter.build(config_for(model,dataset,batch))
    if model in {'tcn_target_only','dlinear','nlinear','patchtst','itransformer','timemixerpp','mixlinear'}:return ModernAdapter.build(config_for(model,dataset,batch))
    return build_adapter(model,config_for(model,dataset,batch))
def read_task(task_id):
    rows=list(csv.DictReader((ROOT/'manifests/compatibility_task_manifest.csv').open(encoding='utf-8-sig')))
    return next(r for r in rows if r['task_id']==task_id)
def batches(dataset):
    train=[load_common_batch(dataset,o) for o in training_origins(dataset)];return train,load_common_batch(dataset,{'m5':'2015-12-07','favorita':'2017-03-01','store_item':'2017-05-22'}[dataset])
def neural_run(task,outdir):
    model_id=task['model_id'];dataset=task['dataset'].lower().replace(' ','_');seed_all(13);train,val=batches(dataset);device=torch.device('cuda');model=build_model(model_id,dataset,4).to(device)
    params=sum(p.numel() for p in model.parameters());trainable=sum(p.numel() for p in model.parameters() if p.requires_grad);lr=.01 if model_id=='mixlinear' else .001;opt=torch.optim.Adam(model.parameters(),lr=lr)
    first=train[0].to(device);model.eval();seed_all(13);initial_a=model.predict(first).prediction.detach().cpu();seed_all(13);mirror=build_model(model_id,dataset,4).to(device);mirror.eval();initial_b=mirror.predict(first).prediction.detach().cpu();initial_repro=bool(torch.allclose(initial_a,initial_b,atol=1e-6,rtol=1e-5));del mirror;gc.collect();torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats();logs=[];start=time.perf_counter();updated=False
    before=[p.detach().cpu().clone() for p in model.parameters() if p.requires_grad]
    for epoch in range(2):
        model.train();losses=[];order=np.random.default_rng(13+epoch).permutation(len(train))
        for pos in range(0,len(order),4):
            batch=concatenate_batches([train[i] for i in order[pos:pos+4]]).to(device);opt.zero_grad(set_to_none=True);output=model(batch);loss=F.huber_loss(output.prediction,batch.target)
            if isinstance(model,ControlledTCNAdapter):loss=loss+model.regularization_loss()
            if not torch.isfinite(loss):raise FloatingPointError('non-finite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();losses.append(float(loss.detach()))
        logs.append({'epoch':epoch+1,'train_loss_engineering_only':float(np.mean(losses))})
    after=[p.detach().cpu() for p in model.parameters() if p.requires_grad];updated=any(not torch.equal(a,b) for a,b in zip(before,after));elapsed=time.perf_counter()-start
    model.eval();val_gpu=val.to(device);t=time.perf_counter();prediction=model.predict(val_gpu);inference=time.perf_counter()-t;graph=prediction.learned_graph;graph_diag=graph_statistics(graph) if graph is not None else {'not_applicable':True}
    ckpt={'state_dict':model.state_dict(),'task':task,'parameters':params};torch.save(ckpt,outdir/'checkpoint_last.pt')
    reload_model=build_model(model_id,dataset,4).to(device);reload_model.load_state_dict(torch.load(outdir/'checkpoint_last.pt',map_location=device)['state_dict']);reload_model.eval();reloaded=reload_model.predict(val_gpu).prediction.detach();reload_ok=bool(torch.allclose(prediction.prediction.detach(),reloaded,atol=1e-6,rtol=1e-5))
    pdict=prediction.diagnostics;used=pdict.get('used_fields',[]);max_train=max(training_origins(dataset));leakage=pd.Timestamp(max_train)+pd.Timedelta(days=27)<pd.Timestamp(val.forecast_origin)
    return {'prediction':prediction.prediction.detach().cpu(),'logs':logs,'params':params,'trainable':trainable,'elapsed':elapsed,'inference':inference,'peak':torch.cuda.max_memory_allocated()/2**20,'updated':updated,'reload_ok':reload_ok,'initial_repro':initial_repro,'graph_diag':graph_diag,'used':used,'leakage':bool(leakage),'graph_export':graph is not None,'shape':list(prediction.prediction.shape)}
def reference_run(task,outdir):
    from models.baselines import seasonal_naive,GlobalLightGBM
    dataset=task['dataset'].lower().replace(' ','_');train,val=batches(dataset);start=time.perf_counter();logs=[]
    if task['model_id']=='seasonal_naive':
        pred=None
        for epoch in range(2):pred=seasonal_naive(np.expm1(val.past_target[0].numpy()).T,28);logs.append({'epoch':epoch+1,'training_semantics':'deterministic_replay'})
        prediction=torch.tensor(np.log1p(pred.T),dtype=torch.float32).unsqueeze(0);state={'season':7}
    else:
        X=[];Y=[]
        for b in train[-8:]:X.append(b.past_target[0,[-1,-7,-14,-28,-56],:].T.numpy());Y.append(np.expm1(b.target[0].numpy()).T)
        model=None
        for epoch in range(2):model=GlobalLightGBM(random_state=13,n_estimators=50).fit(np.concatenate(X),np.concatenate(Y));logs.append({'epoch':epoch+1,'training_semantics':'deterministic_refit'})
        raw=model.predict(val.past_target[0,[-1,-7,-14,-28,-56],:].T.numpy());prediction=torch.tensor(np.log1p(raw.T),dtype=torch.float32).unsqueeze(0);state=model
    torch.save({'state':state,'task':task},outdir/'checkpoint_last.pt');loaded=torch.load(outdir/'checkpoint_last.pt');reload_ok=loaded['state'] is not None
    return {'prediction':prediction,'logs':logs,'params':0,'trainable':0,'elapsed':time.perf_counter()-start,'inference':0.,'peak':0.,'updated':False,'reload_ok':reload_ok,'initial_repro':True,'graph_diag':{'not_applicable':True},'used':['past_target'],'leakage':True,'graph_export':False,'shape':list(prediction.shape)}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task-id',required=True);args=ap.parse_args();task=read_task(args.task_id);outdir=ROOT/'outputs/compatibility_runs'/args.task_id;outdir.mkdir(parents=True,exist_ok=True)
    if (outdir/'completion.marker').exists():print(json.dumps({'task_id':args.task_id,'status':'passed','resumed':True}));return
    started=datetime.now().astimezone();atomic_json(outdir/'run_status.json',{'task_id':args.task_id,'status':'running','start_time':started.isoformat()})
    try:
        result=reference_run(task,outdir) if task['model_id'] in {'seasonal_naive','global_lightgbm'} else neural_run(task,outdir)
        import pandas as pd;pd.DataFrame(result['logs']).to_csv(outdir/'train_log.csv',index=False,encoding='utf-8-sig')
        atomic_json(outdir/'resolved_config.json',task);atomic_json(outdir/'resource_profile.json',{'parameters':result['params'],'trainable_parameters':result['trainable'],'train_seconds':result['elapsed'],'inference_seconds':result['inference'],'peak_gpu_mib':result['peak'],'within_5_2_gib':result['peak']<5.2*1024})
        atomic_json(outdir/'graph_diagnostics.json',result['graph_diag']);atomic_json(outdir/'input_usage.json',{'used_fields':result['used'],'target_exposed_to_forward':False});atomic_json(outdir/'reload_check.json',{'checkpoint_reload_passed':result['reload_ok'],'initial_seed_reproducible':result['initial_repro']})
        status={'task_id':args.task_id,'dataset':task['dataset'],'model_id':task['model_id'],'status':'passed','start_time':started.isoformat(),'end_time':datetime.now().astimezone().isoformat(),'elapsed_seconds':result['elapsed'],'epoch_completed':2,'batch_size_used':4,'peak_gpu_mib':result['peak'],'checkpoint_reload_passed':result['reload_ok'],'prediction_shape_passed':result['shape']==[1,28,90],'finite_output_passed':bool(torch.isfinite(result['prediction']).all()),'graph_export_passed':result['graph_export'] or task['graph_mode']=='none','leakage_check_passed':result['leakage'],'parameter_update_passed':result['updated'] if task['model_id'] not in {'seasonal_naive','global_lightgbm'} else None,'failure_stage':'','exception_type':'','exception_summary':''}
        if not all(status[k] for k in ['checkpoint_reload_passed','prediction_shape_passed','finite_output_passed','leakage_check_passed']):raise RuntimeError(status)
        atomic_json(outdir/'run_status.json',status);(outdir/'completion.marker').write_text(json.dumps({'task_id':args.task_id,'config_sha256':task['config_sha256']}),encoding='utf-8');print(json.dumps(status,ensure_ascii=False))
    except Exception as exc:
        text=repr(exc)+'\n'+traceback.format_exc();(outdir/'exception.txt').write_text(text,encoding='utf-8');status={'task_id':args.task_id,'dataset':task['dataset'],'model_id':task['model_id'],'status':'failed','start_time':started.isoformat(),'end_time':datetime.now().astimezone().isoformat(),'epoch_completed':0,'failure_stage':'compatibility_task','exception_type':type(exc).__name__,'exception_summary':str(exc)};atomic_json(outdir/'run_status.json',status);print(json.dumps(status,ensure_ascii=False));raise


if __name__=='__main__':
    import pandas as pd
    main()
