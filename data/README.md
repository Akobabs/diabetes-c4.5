# Pima Indians Diabetes dataset

Downloaded on 2026-09-05 from [OpenML dataset 37, version 1](https://www.openml.org/d/37), which identifies the source as the UCI Pima Indians Diabetes Database. This is the recommended dataset for the proposal.

- Original download: https://openml.org/data/v1/download/37/diabetes.arff
- Metadata: https://www.openml.org/api/v1/json/data/37
- Original file: `raw/pima_diabetes.arff`
- Saved metadata: `raw/pima_openml_metadata.json`
- Python-friendly CSV: `derived/pima_diabetes.csv`
- OpenML licence field: `Public` (retained verbatim; see saved metadata for attribution).
- Original owners listed in the source: National Institute of Diabetes and Digestive and Kidney Diseases.
- Research citation: Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. Proceedings of the Symposium on Computer Applications and Medical Care, 261–265.

## Verified contents

768 records, eight numeric predictors and one binary outcome; 500 negative and 268 positive outcomes. All 768 converted data rows are distinct. The source describes female participants of Pima Indian heritage aged at least 21. The label is the dataset's existing diabetes classification, not a newly derived threshold or a specified future prediction horizon.

| Source column | CSV column | Meaning / units |
|---|---|---|
| preg | Pregnancies | Number of pregnancies |
| plas | Glucose | Two-hour oral glucose tolerance test plasma glucose; source description uses mg/dl |
| pres | BloodPressure | Diastolic blood pressure, mm Hg |
| skin | SkinThickness | Triceps skinfold thickness, mm |
| insu | Insulin | Two-hour serum insulin, micro-units/ml |
| mass | BMI | Body mass index, kg/m² |
| pedi | DiabetesPedigreeFunction | Diabetes pedigree function |
| age | Age | Years |
| class | Outcome | tested_negative → 0; tested_positive → 1 |

Conversion preserved row order and the eight original numeric values, expanded column names, and mapped the two class labels to 0/1. No imputation, outlier removal, scaling, resampling or train/test splitting has been applied. The ARFF remains unchanged.

The source reports no explicit missing values, but the proposal treats zero values in selected physiological fields as missing indicators: Glucose 5, BloodPressure 35, SkinThickness 227, Insulin 374, and BMI 11. Handle these during fold-specific preprocessing; zero pregnancies remains valid.

## Integrity

The downloaded ARFF's MD5 matches the OpenML metadata: `3cbaa3e54586aa88cf6aacb4033e4470`.

- ARFF SHA-256: `4EDDD5B2B64679E8888348E306520A393D6A28E1DDC9643CFB76FC5D912D6D40`
- CSV SHA-256: `D765AA828A47E8D3BA2F4DE925891BE77803FD0E9568D7C213CDF10C8E83D0B2`

Use `derived/pima_diabetes.csv` for Python training. The original root-level `diabetes.csv` and `diabetes1.csv` are duplicate copies of a different 403-record dataset and must not be mixed with this dataset.
