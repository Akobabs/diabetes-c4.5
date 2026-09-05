"""Check dataset separation, saved predictions, guides, and preservation."""
import json
import hashlib
import zipfile
import pytest
from streamlit.testing.v1 import AppTest
from diabetes_c45.paths import ROOT
from diabetes_c45.external_data import load_external, grouped_holdout
from diabetes_c45.ui_guides import descriptions

@pytest.mark.parametrize('key,rows', [('sylhet',520),('cdc',253680)])
def test_dataset_contract_and_groups(key,rows):
    if not (ROOT/f'data/external/uci_{529 if key=="sylhet" else 891}/data.csv').exists():
        pytest.skip('Additional dataset not downloaded')
    x,y,groups,audit=load_external(key)
    assert len(x)==rows
    assert 'ID' not in x and 'Diabetes_binary' not in x and 'class' not in x
    assert set(descriptions(key,audit))==set(x.columns)
    parts=grouped_holdout(y,groups)
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not set(groups.iloc[parts[a]]) & set(groups.iloc[parts[b]])

@pytest.mark.parametrize('key,count', [('pima',8),('sylhet',16),('cdc',21)])
def test_category_form_and_popup(key,count):
    if key!='pima' and not (ROOT/f'artifacts/external/expansion-20260905/{key}/manifest.json').exists():
        pytest.skip('Additional model not trained')
    app=AppTest.from_file(str(ROOT/'app/streamlit_app.py'),default_timeout=60).run()
    app.sidebar.selectbox(key='prediction_category').set_value(key).run()
    assert not app.exception
    fields=list(app.number_input)+[w for w in app.selectbox if w.key and w.key.startswith(key+'_')]
    assert len(fields)==count
    if key!='pima':
        field=next(w for w in app.selectbox if w.key==('sylhet_gender' if key=='sylhet' else 'cdc_Sex'))
        field.set_value('Female')
        next(b for b in app.button if b.label=='Show research prediction').click().run()
        assert not app.exception
        assert app.metric
        for view in ['Results','Tree and data','Try this model']:
            app.sidebar.radio[0].set_value(view).run()
            assert not app.exception
    next(b for b in app.button if b.key==key+'_guide').click().run()
    assert not app.exception
    assert any('Field-by-field instructions' in m.value for m in app.markdown)

def test_pima_snapshot_and_original_outputs_unchanged():
    folder=ROOT/'preserved/pima-20260905-211750'
    if not folder.exists(): pytest.skip('Local preservation snapshot absent')
    hashes=json.loads((folder/'sha256.json').read_text())
    with zipfile.ZipFile(folder/'pima_snapshot.zip') as archive:
        for name,digest in hashes.items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==digest
            if name.startswith(('artifacts/final/','results/full/','data/raw/','data/derived/')):
                assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
