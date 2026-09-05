"""Additional experiments, separate from the original Pima demonstration."""
import json
import pandas as pd
import streamlit as st
from diabetes_c45.paths import ROOT
from diabetes_c45.external_data import DATASETS
from diabetes_c45.external_predict import ExternalPredictionService
from diabetes_c45.ui_guides import CATEGORIES, ORDINAL, descriptions, field_label, guide_button

if not globals().get('EMBEDDED'):
    st.set_page_config(page_title='Diabetes | Additional studies',page_icon='📊',layout='wide')
st.title('Additional diabetes studies')
st.caption('Separate dataset-specific models. These experiments do not establish external validity of the original Pima model.')
paths=sorted((ROOT/'results/external').glob('*/*/summary.json'),reverse=True)
completed=[]
for path in paths:
    data=json.loads(path.read_text())
    if data.get('status')=='complete' and (not globals().get('FORCED_DATASET') or data['dataset']==FORCED_DATASET):completed.append((path,data))
if not completed:
    st.info('Additional experiments are still running. Completed runs will appear after refreshing this page.')
    st.stop()
selected=st.sidebar.selectbox('Dataset and saved run',range(len(completed)),format_func=lambda i:f"{DATASETS[completed[i][1]['dataset']]['title']} · {completed[i][1]['run_id']}")
path,summary=completed[selected]
folder=path.parent
key=summary['dataset'];run_id=summary['run_id'];spec=DATASETS[key]
audit=json.loads((folder/'audit.json').read_text())
st.sidebar.info(spec['scope'])
st.sidebar.write('Positive label: '+spec['label'])
st.sidebar.caption('Original Pima prototype: run start.bat. This dashboard reads only additional-study artifacts.')
tab=st.sidebar.radio('View',['Try this model','Results','Tree and data'])


@st.cache_resource
def load_service(run_id,key):
    return ExternalPredictionService(run_id,key)


if tab=='Results':
    st.header(spec['title'])
    a,b,c=st.columns(3)
    a.metric('Downloaded records',f"{summary['records']:,}")
    b.metric('Distinct predictor profiles',f"{audit['unique_profiles']:,}")
    c.metric('Held-out predictions per model',f"{summary['evaluated_records']:,}")
    if key=='sylhet':
        st.write('Five outer folds with three inner folds; identical predictor profiles remain together. The final demonstration model is refitted on all records after evaluation.')
    else:
        st.write('Separate training, validation and test profile groups. Parameters are chosen on validation only; the saved model is refitted on training plus validation and evaluated on the reserved test set.')
    st.warning(f"{audit['repeated_profile_rows']:,} repeated-profile rows were kept in the same partition as their matching profiles. {audit['profiles_with_conflicting_labels']:,} profiles have conflicting labels; these were retained together.")
    table=pd.DataFrame({k:{m:v for m,v in values.items() if m!='bootstrap_95'} for k,values in summary['models'].items()}).T
    st.dataframe(table[['accuracy','balanced_accuracy','precision','sensitivity','specificity','f1','roc_auc','average_precision']].style.format('{:.3f}'),width='stretch')
    st.caption(f"Average precision is compared with held-out positive prevalence {summary['positive_prevalence']:.3f}. Class-weighted scores are not calibrated clinical probabilities. Cross-dataset percentages are not directly comparable.")
    st.image(str(folder/'curves.png'))
    with st.expander('Uncertainty and subgroup checks'):
        st.json({name:values['bootstrap_95'] for name,values in summary['models'].items()})
        st.caption('95% profile-bootstrap intervals use 100 resamples of fixed predictions. They exclude model-refitting uncertainty and the CDC survey design.')
        st.dataframe(pd.DataFrame(json.loads((folder/'subgroups.json').read_text())),width='stretch')
    st.image(str(folder/'j48_confusion.png'))
    st.download_button('Download study summary',path.read_bytes(),f'{key}_summary.json','application/json')
elif tab=='Try this model':
    service=load_service(run_id,key)
    st.header(CATEGORIES[key])
    st.info('Use only the measurement definitions from this dataset. This demonstration does not establish a clinical diagnosis.')
    hints=descriptions(key,audit)
    guide_button(key,audit)
    with st.form(f'{run_id}_{key}'):
        values={};columns=st.columns(2)
        for i,col in enumerate(service.manifest['features']):
            with columns[i%2]:
                choices=audit['categories'].get(col)
                if key=='cdc' and col in ORDINAL:
                    selected_value=st.selectbox(field_label(col),['Unknown']+ORDINAL[col],help=hints.get(col),key=f'{key}_{col}')
                    values[col]=None if selected_value=='Unknown' else ORDINAL[col].index(selected_value)+1
                elif choices:
                    display_choices=(['Female','Male'] if col=='Sex' else ['No','Yes']) if key=='cdc' else choices
                    selected_value=st.selectbox(field_label(col),['Unknown']+display_choices,help=hints.get(col),key=f'{key}_{col}')
                    values[col]=None if selected_value=='Unknown' else display_choices.index(selected_value)
                else:
                    low,high=0.0,None
                    if key=='cdc':
                        ranges={'Age':(1.,13.),'Education':(1.,6.),'Income':(1.,8.),'GenHlth':(1.,5.),'MentHlth':(0.,30.),'PhysHlth':(0.,30.)}
                        low,high=ranges.get(col,(0.,None))
                    integer=col in ['age','MentHlth','PhysHlth']
                    values[col]=st.number_input(field_label(col),min_value=int(low) if integer else low,max_value=int(high) if integer and high is not None else high,value=None,step=1 if integer else 0.1,help=hints.get(col),key=f'{key}_{col}')
        submitted=st.form_submit_button('Show research prediction',type='primary')
    if submitted:
        if all(v is None for v in values.values()):
            st.warning('Enter at least one predictor.')
        else:
            try:
                result=service.predict(values)
                st.subheader(result['label']);st.metric('Positive-class score',f"{result['score']:.3f}")
                if result['imputed']:st.info('Training-only imputation used for: '+', '.join(result['imputed']))
                st.write('Actual J48 decision path')
                for step in result['steps']:st.write(f"{step['feature']}: {step['value']} {step['operator']} {step['threshold']}")
                st.download_button('Download result',json.dumps(result,indent=2),f'{key}_prediction.json','application/json')
            except ValueError as exc:st.error(str(exc))
else:
    model_dir=ROOT/'artifacts/external'/run_id/key
    st.header('Tree and source data')
    st.graphviz_chart((model_dir/'tree.dot').read_text())
    st.json(summary['final_params'])
    st.markdown(f"[UCI dataset documentation]({audit['source']})")
    st.dataframe(pd.DataFrame(audit['variables']),width='stretch')
    st.caption('Age in the CDC dataset is an ordinal band; it cannot be substituted for Pima or Sylhet age in years. Binary indicators cannot substitute for glucose or blood pressure measurements.')
