"""Versioned, separate diabetes experiments with duplicate-profile isolation."""
import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .paths import ROOT
from .data import sha256
from .evaluate import metrics, write_json
from .external_data import DATASETS, load_external, grouped_holdout, ExternalPreprocessor
from .external_j48 import ExternalJ48

NAMES = ['J48', 'Naive Bayes', 'Logistic Regression']


def score_metrics(y, score):
    result=metrics(y,score)
    result.update(average_precision=float(average_precision_score(y,score)),
                  balanced_accuracy=float(balanced_accuracy_score(y,np.asarray(score)>=.5)),
                  brier_score=float(brier_score_loss(y,score)))
    return result


def candidate_list(name, cfg, global_cfg):
    if name=='J48':
        return [dict(confidence=c,min_leaf=m,balanced=b) for c,m,b in itertools.product(cfg['confidence'],cfg['min_leaf'],cfg['balanced'])]
    if name=='Naive Bayes':
        return [dict(smoothing=s,balanced=b) for s,b in itertools.product(global_cfg['baselines']['naive_bayes_smoothing'],[False,True])]
    return [dict(c=c,balanced=b) for c,b in itertools.product(global_cfg['baselines']['logistic_c'],[False,True])]


def fit_model(name,params,x,y,categories):
    if name=='J48':
        return ExternalJ48(categories,**params).fit(x,y)
    if name=='Naive Bayes':
        weights=len(y)/(2*np.bincount(y))[np.asarray(y)] if params['balanced'] else None
        return GaussianNB(var_smoothing=params['smoothing']).fit(x,y,sample_weight=weights)
    return make_pipeline(StandardScaler(),LogisticRegression(C=params['c'],class_weight='balanced' if params['balanced'] else None,max_iter=2000)).fit(x,y)


def select_model(name,x,y,groups,splits,categories,cfg,global_cfg):
    prepared=[]
    for tr,va in splits:
        if set(groups.iloc[tr]) & set(groups.iloc[va]): raise ValueError('Duplicate profile leakage')
        if y.iloc[tr].nunique()!=2 or y.iloc[va].nunique()!=2: raise ValueError('A split lacks one class')
        prep=ExternalPreprocessor(categories).fit(x.iloc[tr])
        prepared.append((prep.transform(x.iloc[tr]),y.iloc[tr],prep.transform(x.iloc[va]),y.iloc[va]))
    records=[]
    options=candidate_list(name,cfg,global_cfg)
    for number,params in enumerate(options,1):
        aps,sizes=[],[]
        print(f'  {name}: candidate {number}/{len(options)} {params}',flush=True)
        for xt,yt,xv,yv in prepared:
            model=fit_model(name,params,xt,yt,categories)
            aps.append(float(average_precision_score(yv,model.predict_proba(xv)[:,1])))
            sizes.append(model.tree_size if name=='J48' else 0)
        records.append(dict(params=params,mean_validation_average_precision=float(np.mean(aps)),mean_tree_size=float(np.mean(sizes))))
    best=min(records,key=lambda r:(-r['mean_validation_average_precision'],r['mean_tree_size']))
    return best['params'],records


def confidence_intervals(y,score,groups,seed):
    # Bootstrap whole profiles, conditional on saved predictions. This does not
    # quantify refitting uncertainty, CV dependence, or the CDC survey design.
    codes,_=pd.factorize(groups)
    count=int(codes.max())+1
    rng=np.random.default_rng(seed)
    values={'roc_auc':[],'average_precision':[]}
    for _ in range(100):
        weights=np.bincount(rng.integers(0,count,count),minlength=count)[codes]
        if np.unique(np.asarray(y)[weights>0]).size<2: continue
        values['roc_auc'].append(roc_auc_score(y,score,sample_weight=weights))
        values['average_precision'].append(average_precision_score(y,score,sample_weight=weights))
    return {k:[float(q) for q in np.quantile(v,[.025,.975])] for k,v in values.items()}


def save_model(model,prep,key,run_id,audit,params):
    folder=ROOT/'artifacts/external'/run_id/key
    folder.mkdir(parents=True,exist_ok=False)
    model.save(folder)
    joblib.dump(prep,folder/'preprocessing.joblib')
    write_json(folder/'tree.json',model.export_tree())
    manifest=dict(dataset=key,run_id=run_id,model='WEKA J48 C4.5',params=params,
                  features=prep.features_,categories=prep.categories,medians=prep.fill_.to_dict(),
                  source_sha256=audit['sha256'],tree_size=model.tree_size,leaf_count=model.leaf_count,
                  label=DATASETS[key]['label'],negative=DATASETS[key]['negative'],scope=DATASETS[key]['scope'],
                  threshold=.5,created_utc=datetime.now(timezone.utc).isoformat(),
                  files={n:sha256(folder/n) for n in ['j48.model','preprocessing.joblib','tree.json','tree.txt','tree.dot']})
    write_json(folder/'manifest.json',manifest)
    return manifest


def subgroup_report(x,y,score,key):
    sex='gender' if key=='sylhet' else 'Sex'
    age='age' if key=='sylhet' else 'Age'
    pivot=45 if key=='sylhet' else 9
    age_names=['Age under 45','Age 45+'] if key=='sylhet' else ['Age category 1-8 (under 60)','Age category 9-13 (60+)']
    masks={'Female':x[sex].eq(0),'Male':x[sex].eq(1),age_names[0]:x[age].lt(pivot),age_names[1]:x[age].ge(pivot)}
    rows=[]
    for label,mask in masks.items():
        if mask.sum() and y[mask].nunique()==2:
            rows.append(dict(group=label,n=int(mask.sum()),positives=int(y[mask].sum()),**score_metrics(y[mask],np.asarray(score)[mask])))
    return rows


def figures(frame,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve,roc_curve,ConfusionMatrixDisplay
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,rows in frame.groupby('model'):
        fpr,tpr,_=roc_curve(rows.outcome,rows.score)
        precision,recall,_=precision_recall_curve(rows.outcome,rows.score)
        axes[0].plot(fpr,tpr,label=name)
        axes[1].plot(recall,precision,label=name)
    axes[0].plot([0,1],[0,1],'--',color='gray')
    axes[0].set(xlabel='False positive rate',ylabel='Sensitivity',title='Held-out ROC')
    prevalence=frame[frame.model.eq('J48')].outcome.mean()
    axes[1].axhline(prevalence,linestyle='--',color='gray',label='Prevalence baseline')
    axes[1].set(xlabel='Recall',ylabel='Precision',title='Held-out precision-recall')
    for ax in axes:ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(folder/'curves.png',dpi=180);plt.close(fig)
    rows=frame[frame.model.eq('J48')]
    fig,ax=plt.subplots(figsize=(4,4))
    ConfusionMatrixDisplay.from_predictions(rows.outcome,rows.score.ge(.5),display_labels=['Negative','Positive'],ax=ax,colorbar=False)
    fig.tight_layout();fig.savefig(folder/'j48_confusion.png',dpi=180);plt.close(fig)


def run(key,run_id,global_cfg):
    folder=ROOT/'results/external'/run_id/key
    folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/'summary.json',{'status':'running','dataset':key})
    x,y,groups,audit=load_external(key)
    write_json(folder/'audit.json',audit)
    write_json(folder/'config.json',global_cfg)
    x.describe().to_csv(folder/'descriptive_statistics.csv')
    cfg=global_cfg[key]; seed=global_cfg['seed']; categories=audit['categories']
    print(f'{key}: {len(x)} records, {groups.nunique()} distinct predictor profiles',flush=True)
    predictions=[];fold_metrics=[];final_params={}
    if key=='sylhet':
        outer=list(StratifiedGroupKFold(cfg['outer_folds'],shuffle=True,random_state=seed).split(x,y,groups))
        write_json(folder/'folds.json',[dict(train=tr.tolist(),test=te.tolist()) for tr,te in outer])
        for fold,(tr,te) in enumerate(outer,1):
            if set(groups.iloc[tr]) & set(groups.iloc[te]):raise ValueError('Outer group leakage')
            xt,yt,gt=x.iloc[tr],y.iloc[tr],groups.iloc[tr]
            inner=list(StratifiedGroupKFold(cfg['inner_folds'],shuffle=True,random_state=seed+fold).split(xt,yt,gt))
            write_json(folder/f'inner_folds_{fold}.json',[dict(train=tr[a].tolist(),validation=tr[b].tolist()) for a,b in inner])
            for name in NAMES:
                print(f'{key}: outer fold {fold}/{len(outer)} {name}',flush=True)
                params,search=select_model(name,xt,yt,gt,inner,categories,cfg,global_cfg)
                write_json(folder/f'search_{fold}_{name.replace(" ","_")}.json',search)
                prep=ExternalPreprocessor(categories).fit(xt)
                model=fit_model(name,params,prep.transform(xt),yt,categories)
                scores=model.predict_proba(prep.transform(x.iloc[te]))[:,1]
                predictions.extend(dict(model=name,fold=fold,row_id=int(i),outcome=int(y.iloc[i]),score=float(s),profile=groups.iloc[i]) for i,s in zip(te,scores))
                fold_metrics.append(dict(model=name,fold=fold,params=json.dumps(params),**score_metrics(y.iloc[te],scores)))
        final_splits=list(StratifiedGroupKFold(cfg['inner_folds'],shuffle=True,random_state=seed).split(x,y,groups))
        final_params['J48'],search=select_model('J48',x,y,groups,final_splits,categories,cfg,global_cfg)
        write_json(folder/'final_search.json',search)
        development=np.arange(len(x))
    else:
        partitions=grouped_holdout(y,groups,seed)
        write_json(folder/'splits.json',{k:v.tolist() for k,v in partitions.items()})
        tr,va,te=partitions['train'],partitions['validation'],partitions['test']
        development=np.concatenate([tr,va])
        for name in NAMES:
            print(f'{key}: validation tuning {name}; train={len(tr)}, validation={len(va)}, test={len(te)}',flush=True)
            params,search=select_model(name,x,y,groups,[(tr,va)],categories,cfg,global_cfg)
            final_params[name]=params
            write_json(folder/f'search_{name.replace(" ","_")}.json',search)
            prep=ExternalPreprocessor(categories).fit(x.iloc[development])
            model=fit_model(name,params,prep.transform(x.iloc[development]),y.iloc[development],categories)
            scores=model.predict_proba(prep.transform(x.iloc[te]))[:,1]
            predictions.extend(dict(model=name,fold=0,row_id=int(i),outcome=int(y.iloc[i]),score=float(s),profile=groups.iloc[i]) for i,s in zip(te,scores))
            fold_metrics.append(dict(model=name,fold=0,params=json.dumps(params),**score_metrics(y.iloc[te],scores)))
            if name=='J48':
                manifest=save_model(model,prep,key,run_id,audit,params)
        # The CDC demo keeps the evaluated development-only model. Test labels
        # are never used to refit, select a threshold, or improve this model.
    frame=pd.DataFrame(predictions)
    frame.to_csv(folder/'predictions.csv',index=False)
    pd.DataFrame(fold_metrics).to_csv(folder/'fold_metrics.csv',index=False)
    if key=='sylhet':
        prep=ExternalPreprocessor(categories).fit(x)
        model=fit_model('J48',final_params['J48'],prep.transform(x),y,categories)
        manifest=save_model(model,prep,key,run_id,audit,final_params['J48'])
    models={}
    for name,rows in frame.groupby('model'):
        models[name]=score_metrics(rows.outcome,rows.score)
        models[name]['bootstrap_95']=confidence_intervals(rows.outcome,rows.score,rows.profile,seed)
    jr=frame[frame.model.eq('J48')].sort_values('row_id')
    subgroups=subgroup_report(x.iloc[jr.row_id],y.iloc[jr.row_id],jr.score.to_numpy(),key)
    write_json(folder/'subgroups.json',subgroups)
    figures(frame,folder)
    summary=dict(status='complete',dataset=key,run_id=run_id,source_sha256=audit['sha256'],protocol=cfg,
                 records=len(x),evaluated_records=len(jr),positive_prevalence=float(jr.outcome.mean()),
                 models=models,final_params=final_params,tree_size=manifest['tree_size'],leaf_count=manifest['leaf_count'],
                 model_training_rows=len(development),target=DATASETS[key]['label'],scope=DATASETS[key]['scope'],
                 limitations=['Separate dataset-specific experiments, not external validation of the original Pima model.',
                              'No clinical deployment or prospective validation claim.',
                              'Profile-group bootstrap intervals condition on saved predictions; they exclude model-refitting uncertainty and survey-design effects.',
                              'Class-weighted tree scores are not calibrated probabilities.',
                              'Subgroup figures are descriptive; small subgroups can be unstable.'])
    write_json(folder/'summary.json',summary)
    print(f'Completed {key}: '+json.dumps(models['J48']),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['sylhet','cdc','all'],default='all')
    parser.add_argument('--run-id',default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    args=parser.parse_args()
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id):
        raise ValueError('Run ID must contain letters, digits, underscores or hyphens only')
    cfg=json.loads((ROOT/'config/external_experiments.json').read_text())
    for key in (DATASETS if args.dataset=='all' else [args.dataset]):run(key,args.run_id,cfg)
    print('Preserved Pima outputs; new run:',args.run_id,flush=True)


if __name__=='__main__':main()
