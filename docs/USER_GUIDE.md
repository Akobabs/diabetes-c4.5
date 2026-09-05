# Using the research prototype

## Choose an assessment

Double-click `start.bat`. The sidebar's **Prediction category** selects clinical measurements (Pima), symptoms (Sylhet), or health indicators (CDC). Each category loads its own model and form. Use **How to fill this assessment** above the form to open a detailed guide, and the help icon beside a field for its definition.

The Pima category concerns diabetes in its original women's study population. It does not assess pregnancy or gestational diabetes. Its pregnancy-count field describes history only. Use the specified laboratory tests and units; leave unavailable values blank instead of guessing.

For Sylhet, answer symptom questions with Yes, No or Unknown. For CDC, follow the stated reporting periods and choose named age, health, education and income categories. These choices map to dataset codes automatically. CDC's positive outcome includes both prediabetes and diabetes.

Submit the form to see the saved model's classification, score and decision path. Any filled-in missing values are disclosed. A model score is not a calibrated medical risk and a negative result cannot rule out diabetes. These studies are separate; comparing their scores for one person does not create a combined assessment.

For the new categories, choose **Results** or **Tree and data** in the sidebar to inspect held-out evaluation, limitations, source definitions and the saved tree. See [the additional-study report](ADDITIONAL_DATASET_RESULTS.md) for evaluation protocols and preservation details.

## Start

In the project directory run:

```powershell
.venv/Scripts/python.exe -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

Open the displayed local address, normally http://127.0.0.1:8501. Keep the terminal running; Ctrl+C stops the application when it was launched in that terminal.

## Make a prediction

1. Choose **Predict** in the sidebar.
2. Enter the clinical measurements using the units shown. Glucose and insulin refer to the two-hour test measurements; the blood-pressure field is diastolic pressure.
3. Leave unknown values blank. At least one measurement is required in the form. Zero glucose, pressure, skin thickness, insulin and BMI are handled as unknown, following the dataset preprocessing protocol. Zero pregnancies remains a valid value.
4. Select **Show prediction**.
5. Read the predicted class, model score, any imputed or capped inputs, and the decision path. Editing the form does not update the previous result until it is submitted again.
6. Optionally download the result as JSON. The file includes full-precision split thresholds and the preprocessing changes for this submission.

The application uses medians and bounds learned when the final model was trained. It does not learn from the submitted patient. If an input is outside the observed training range, the interface points that out. An unknown input can still be important to the decision; extensive imputation limits how informative the result is.

## Read the results

Screenshots from the working prototype are saved in [screenshots/](screenshots/): [input form](screenshots/prediction_form.png), [example prediction](screenshots/prediction_result.png), [research results](screenshots/research_results.png), and [decision tree](screenshots/decision_tree.png). The example uses a record from the research dataset.

- **Research results** shows held-out scores for C4.5, Naive Bayes and Logistic Regression, confusion matrices and ROC curves. The expansion panels show fold variability and fixed preprocessing comparisons.
- **Decision tree** displays the saved final J48 tree, the retained predictors and downloadable text rules.
- **About the study** explains the dataset and research scope.

Accuracy measures all correct classifications. Sensitivity measures the proportion of positive cases detected, while specificity measures the proportion of negative cases correctly classified. AUC summarises discrimination across thresholds. The app uses a fixed decision threshold of 0.5; its model score is not a calibrated clinical probability.

## Research boundaries

This is a prototype based on women aged 21 or older of Pima Indian heritage. It predicts the dataset's existing binary label. It does not establish a clinical diagnosis, estimate a specific future onset date, or demonstrate accuracy in other populations. No patient database is created; inputs and the last result are held in the current application session, with optional user-initiated downloads.

## Troubleshooting

- **Model preparation:** run the full evaluation and final training commands in README.md.
- **Java cannot be found:** set `JAVA_HOME` to a compatible 64-bit Java installation or use the project-local runtime.
- **Artifact integrity error:** restore the matching model files together or regenerate them with the documented training command.
- **Port already in use:** use the existing server or select another port, for example `--server.port 8502`.
- **After changing dependencies or Java:** restart the Streamlit process so it starts a fresh JVM.
