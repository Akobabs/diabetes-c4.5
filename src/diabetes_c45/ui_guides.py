"""User-facing dataset definitions shared by forms and modal guides."""
import streamlit as st

CATEGORIES = {
    'pima': "Women's diabetes — Clinical measurements (Pima)",
    'sylhet': 'Diabetes study — Symptoms (Sylhet)',
    'cdc': 'Prediabetes / diabetes — Health indicators (CDC)',
}
PIMA = {
    'Pregnancies': 'Number of pregnancies recorded in the medical history; enter a whole number. Zero is valid. This is not number of children, weeks pregnant, or a prediction of pregnancy.',
    'Glucose': 'Plasma glucose measured two hours into an oral glucose tolerance test, in mg/dL. Use the matching laboratory result; do not substitute fasting glucose, HbA1c, or a random glucose reading.',
    'BloodPressure': 'Diastolic blood pressure in mmHg: the bottom number. For a reading of 120/80, enter 80. This example explains the format, not a diagnostic threshold.',
    'SkinThickness': 'Triceps skinfold thickness in millimetres, measured using a skinfold caliper. This is not arm circumference or body-fat percentage. Leave unknown if it was not measured.',
    'Insulin': 'Two-hour serum insulin in micro-units/mL, from the matching laboratory test. This is not an insulin injection dose.',
    'BMI': 'Body mass index in kg/m²: weight in kilograms divided by height in metres squared. Enter BMI, not weight or height.',
    'DiabetesPedigreeFunction': 'The dataset-specific numerical family-history score. It is not a percentage or a count of affected relatives. Use a documented score; this prototype cannot calculate it from family history. Otherwise leave unknown.',
    'Age': 'Age in completed years, as a whole number. The Pima study includes women aged 21 or older.',
}
SYMPTOMS = {
    'age': 'Age in completed years; enter a whole number.',
    'gender': 'Sex/gender category recorded by this study: Female or Male. Select Unknown if these source categories do not provide an appropriate answer.',
    'polyuria': 'Excessive or unusually frequent urination.',
    'polydipsia': 'Unusually increased thirst.',
    'sudden_weight_loss': 'Sudden, unexplained loss of weight.',
    'weakness': 'Reported weakness or lack of strength.',
    'polyphagia': 'Unusually increased hunger.',
    'genital_thrush': 'Reported genital yeast infection; do not guess a diagnosis from irritation alone.',
    'visual_blurring': 'Reported blurred vision.',
    'itching': 'Reported itching.',
    'irritability': 'Reported increased irritability.',
    'delayed_healing': 'Reported slow healing of wounds or sores.',
    'partial_paresis': 'Partial loss of muscle strength or movement (partial paresis).',
    'muscle_stiffness': 'Reported muscle stiffness.',
    'alopecia': 'Reported hair loss (alopecia).',
    'obesity': 'Obesity as recorded in the study questionnaire. Do not invent a BMI cutoff for this field.',
}
LABELS = {
    'polyuria': 'Excessive urination (polyuria)', 'polydipsia': 'Increased thirst (polydipsia)',
    'polyphagia': 'Increased hunger (polyphagia)', 'partial_paresis': 'Partial muscle weakness (paresis)',
    'alopecia': 'Hair loss (alopecia)', 'HighBP': 'High blood pressure',
    'HighChol': 'High cholesterol', 'CholCheck': 'Cholesterol checked within 5 years',
    'Smoker': 'At least 100 cigarettes over your lifetime', 'HeartDiseaseorAttack': 'Coronary heart disease or heart attack',
    'PhysActivity': 'Physical activity outside work', 'Fruits': 'Fruit at least daily',
    'Veggies': 'Vegetables at least daily', 'HvyAlcoholConsump': 'Heavy alcohol consumption (study definition)',
    'AnyHealthcare': 'Healthcare coverage', 'NoDocbcCost': 'Unable to see a doctor because of cost',
    'GenHlth': 'General health', 'MentHlth': 'Days mental health was not good',
    'PhysHlth': 'Days physical health was not good', 'DiffWalk': 'Serious difficulty walking or climbing stairs',
    'BMI': 'Body mass index (kg/m²)', 'Age': 'Age group', 'Sex': 'Recorded sex',
    'Income': 'Annual household income (US dollars)',
}
ORDINAL = {
    'Age': ['18–24', '25–29', '30–34', '35–39', '40–44', '45–49', '50–54', '55–59', '60–64', '65–69', '70–74', '75–79', '80 or older'],
    'GenHlth': ['Excellent', 'Very good', 'Good', 'Fair', 'Poor'],
    'Education': ['Never attended school / kindergarten only', 'Grades 1–8', 'Grades 9–11', 'Grade 12 / GED', 'College 1–3 years / technical school', 'College 4+ years'],
    'Income': ['Below $10,000', '$10,000–14,999', '$15,000–19,999', '$20,000–24,999', '$25,000–34,999', '$35,000–49,999', '$50,000–74,999', '$75,000 or more'],
}

def field_label(name):
    return LABELS.get(name, name.replace('_', ' ').title())

def descriptions(key, audit=None):
    if key == 'pima': return PIMA
    if key == 'sylhet': return SYMPTOMS
    result = {v['name']: v.get('description') or '' for v in audit['variables'] if v['role'] == 'Feature'}
    result['BMI'] = PIMA['BMI']
    result['MentHlth'] += ' Enter a whole number from 0 to 30; 0 means no such days.'
    result['PhysHlth'] += ' Enter a whole number from 0 to 30; 0 means no such days.'
    return result

@st.dialog('How to fill this assessment', width='large')
def show_guide(key, audit=None):
    st.subheader(CATEGORIES[key])
    if key == 'pima':
        st.write('This model classifies the original diabetes outcome in Pima study records. It does not predict pregnancy, fertility, or gestational diabetes. Its study population was women aged 21 or older of Pima Indian heritage.')
        st.write('Use recorded measurements with the specified test and units. Leave unavailable fields blank. Zero pregnancies is valid; zero glucose, blood pressure, skinfold, insulin or BMI is treated as missing by this pipeline. Missing values use saved training medians.')
    else:
        st.write('Select Yes only when the stated condition applies, No when it does not, and Unknown when you cannot answer. Missing entries use saved training medians or category modes. An imputed answer is not a measurement of you.')
        if key == 'sylhet':
            st.write('This questionnaire model was trained on participants at Sylhet Diabetes Hospital, Bangladesh. The symptom fields indicate presence, not severity. The published fields do not establish a common reporting period; do not assume the CDC 30-day window applies here.')
        else:
            st.write('This US survey model uses health history and behaviours. Its positive outcome combines prediabetes and diabetes. Follow each field’s time window. Choose named age, education, income and health categories; the form converts them to the original study codes.')
    st.markdown('### Field-by-field instructions')
    for name, description in descriptions(key, audit).items():
        label = name if key == 'pima' else field_label(name)
        st.markdown(f'**{label}** — {description}')
        if key == 'cdc' and name in ORDINAL:
            st.caption('; '.join(f'{i}: {value}' for i, value in enumerate(ORDINAL[name], 1)))
    st.markdown('### Understanding the result')
    st.write('The classification is the saved C4.5 tree’s output at a score threshold of 0.5. The score is not a calibrated personal medical risk. The decision path explains which recorded inputs the tree used; it does not show what caused diabetes. A negative classification does not rule out diabetes. These separate study models should not be averaged into one prediction.')
    st.markdown('[Diabetes testing and diagnosis (NIDDK)](https://www.niddk.nih.gov/health-information/diabetes/overview/tests-diagnosis) · [Symptoms (NIDDK)](https://www.niddk.nih.gov/health-information/diabetes/overview/symptoms-causes)')
    if audit: st.markdown(f"[Dataset definitions (UCI)]({audit['source']})")
    if key == 'cdc':
        st.markdown('[Age and income coding reference (CDC BRFSS)](https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf)')

def guide_button(key, audit=None):
    if st.button('How to fill this assessment', key=f'{key}_guide'):
        show_guide(key, audit)
