# Diabetes prediction with C4.5

Python training and evaluation, genuine WEKA J48 C4.5, and a Streamlit research prototype. Input: the verified 768-record Pima dataset in `data/derived/pima_diabetes.csv`. The original root CSVs belong to a different dataset and are not used.

The project includes data preparation, nested cross-validation, baseline comparisons, final model training, prediction explanations, and a browser-based prototype. The initial completed implementation passed all 9 tests, including app navigation and reproduction of saved held-out scores. See [the results draft](docs/IMPLEMENTATION_RESULTS.md), [implementation plan](C45_IMPLEMENTATION_PLAN.md), and [user guide](docs/USER_GUIDE.md).

## Quick start on Windows

Two batch files in the project root handle the usual workflows. Double-click the appropriate file in File Explorer, or run it from PowerShell:

| File | Purpose | Retrains? |
|---|---|---|
| [start.bat](start.bat) | Check the existing environment and saved model, then open the Streamlit prototype | No |
| [FullRun.bat](FullRun.bat) | Set up/check dependencies, recreate the dataset, rerun the entire research experiment, train a fresh model, generate reports, run tests, then open Streamlit | Yes |

```powershell
# Demonstrate an existing trained model
.\start.bat

# Build all research outputs and train again from scratch
.\FullRun.bat
```

The launchers locate the project relative to their own files, so paths containing spaces are supported and they do not depend on the terminal's current directory. Keep the launcher window open while using the prototype. Press **Ctrl+C** to stop Streamlit. A window opened by double-click remains open on failure so you can read the error; press Enter to close it.

The default address is **http://127.0.0.1:8501**. If that port is occupied, the launcher chooses an available port among the next nine and prints the actual address. It never kills another server. The browser normally opens automatically; if it does not, copy the printed address into your browser.

### Launcher options

Both batch files accept these optional flags:

| Flag | Behaviour |
|---|---|
| `-CheckOnly` | Validate the installed environment, original data, full research configuration and available port; do not install dependencies, archive outputs, train, or launch Streamlit. `start.bat` also loads the saved model and runs one sample prediction. Logs are still written. |
| `-NoBrowser` | Start Streamlit without automatically opening a browser |
| `-NoPause` | Return immediately when the workflow ends or fails; useful in terminals and automation |
| `-Port 8601` | Try port 8601 first, then the next nine ports if needed |

```powershell
.\start.bat -CheckOnly -NoPause
.\FullRun.bat -CheckOnly -NoPause
.\start.bat -NoBrowser -NoPause -Port 8601
.\FullRun.bat -NoBrowser -NoPause
```

`FullRun.bat -CheckOnly` checks the environment as currently installed. It is not an installation command and does not require an existing trained model. A fresh checkout without `.venv` needs a normal `FullRun.bat` first.

The batch files delegate to [scripts/windows_runner.ps1](scripts/windows_runner.ps1). Execution-policy bypass applies only to that child PowerShell process; the launchers do not change your permanent PowerShell policy. Successful commands return exit code 0; failures return a nonzero code and stop subsequent steps.

## Requirements and first-time setup

- Windows with Windows PowerShell 5.1 or later and a browser.
- **64-bit Python 3.12.** The Windows launchers enforce this version to match the tested environment. If needed, install it from [Python's Windows downloads](https://www.python.org/downloads/windows/), including the Python launcher, then check `py -3.12 --version`.
- **64-bit Java**, tested with Java 17. Python calls WEKA J48 through `python-weka-wrapper3`, so Java is required even though the project code is Python.
- Internet access for the first dependency installation, subsequent missing package downloads, or downloading Java. A prepared `start.bat` run uses local resources.
- Free disk space for the virtual environment, runtime, generated results and backups. Allow several GB; the full model search is CPU work and does not require a GPU.

On a fresh checkout:

1. Install 64-bit Python 3.12 if it is not already available.
2. Ensure the project includes `requirements.lock`, `config/experiment.json`, `data/raw/pima_diabetes.arff`, and `data/raw/pima_openml_metadata.json`.
3. Run **FullRun.bat**. It creates `.venv` when absent and installs the pinned dependencies.
4. If Java is not available through `JAVA_HOME` or `.runtime/java`, FullRun downloads an Eclipse Temurin Java 17 Windows runtime into `.runtime/java`, verifies the publisher-provided SHA-256 checksum, and uses it for this process. An existing usable runtime is reused. See [Eclipse Adoptium](https://adoptium.net/) and the [Python WEKA wrapper installation guide](https://fracpete.github.io/python-weka-wrapper3/install.html).
5. Wait for evaluation, training, reports and tests to finish. Streamlit launches only after all required checks pass.

The launcher searches for Python in a working project `.venv`, project-local `.runtime/python`, the `py -3.12` launcher, and then `python` on PATH. An incompatible existing `.venv` is reported rather than silently deleted; rename that folder and rerun FullRun to create a new environment. No machine-wide Python or Java installation is performed by these scripts. An invalid configured Java runtime must be corrected if it fails JVM startup.

Generated environments and trained model files are excluded from Git. Therefore a fresh clone normally needs **FullRun.bat** before **start.bat**, even if this README shows results from the original completed run.

## What FullRun does from scratch

FullRun repeats the complete training and evaluation procedure. It never uses `--smoke`, previous fitted models, previous fold predictions, or `--tuning-cache`. It reuses a compatible Python/Java installation and the immutable source download; "from scratch" refers to derived data and learned/reported research outputs, rather than reinstalling the operating system or redownloading unchanged source data.

The exact order is:

1. Locate/create the Python environment and locate/download Java. Install `requirements.lock`, reinstall this project in editable mode, and run `pip check`.
2. Check dependency versions, JVM startup, the source ARFF checksum and the required **10 outer / 5 inner folds**. Check that a local frontend port is available.
3. Recreate `data/derived/pima_diabetes.csv` from the preserved ARFF and verify its reference SHA-256 checksum.
4. Test the Windows archive/error-handling helpers with isolated fixture files, then archive previous `results/full`, `artifacts/final`, `docs/IMPLEMENTATION_RESULTS.md` and `docs/screenshots` under `.runtime/backups/<timestamp>/`, preserving their relative paths. Source data and source code are not moved.
5. Run the core data-pipeline and J48 integration tests.
6. Run fresh nested evaluation for J48, Gaussian Naive Bayes and Logistic Regression, plus the fixed J48 comparisons. Save new splits, parameter searches, predictions, metrics and plots.
7. Run a fresh full-data J48 parameter search and fit the final demonstration model; save preprocessing, header, tree and provenance metadata.
8. Generate the research report and analysis figures, run the complete test suite, verify the saved model and evaluation hashes, and perform a sample prediction.
9. Start Streamlit and normally open the browser.

The full run takes considerably longer than launching the prototype. The initial experiment took tens of minutes on this machine; hardware and package installation time can change this. Progress is printed by outer fold and classifier. The full experiment currently evaluates **128 J48 configurations per inner search**, plus baseline configurations and controlled comparisons.

### Logs, backups and interrupted runs

- Every launcher invocation writes a timestamped transcript to `.runtime/logs/`.
- Previous generated results are archived before retraining, so old completed results cannot be mistaken for the new run. Paths are checked to stay inside this workspace; archives through symlinks or junctions are rejected.
- If a command fails, later commands and the frontend launch are stopped. Review the first failure in the transcript, resolve it and run FullRun again. It starts the model work afresh rather than resuming partial search results.
- Backups are retained until you remove them yourself. They are not automatically restored after a failure. If restoring manually, restore the model, matching results, and report together from the same backup; do not mix model and evaluation versions.
- Close older prototype sessions before a rebuild so you do not accidentally demonstrate their cached model. FullRun does not terminate other running applications.
- Figures and the Markdown results report are regenerated automatically. Browser screenshots are model-specific manual documentation: old screenshots are archived and should be recaptured after a rebuild if needed for the dissertation.

## Reference research results

| Model | Held-out accuracy | Sensitivity | ROC-AUC |
|---|---:|---:|---:|
| J48 C4.5 | 74.35% | 75.37% | 0.8117 |
| Naive Bayes | 73.96% | 72.39% | 0.8217 |
| Logistic Regression | 76.69% | 61.94% | 0.8233 |

These are pooled predictions from nested outer validation, not training scores. The final demonstration tree has 13 nodes and 7 leaves. Its selected configuration uses five predictors, confidence factor 0.25, minimum leaf size 20, no SMOTE and no outlier capping.

## Frontend pages and prediction workflow

The frontend is **Streamlit**, written in Python. There is no Node.js, npm, React or TypeScript build step in this implementation.

| Page | What it provides |
|---|---|
| Predict | Eight measurement fields, positive/negative classification, model score, imputation/capping notices and the actual J48 decision path |
| Research results | Held-out model comparison, fold variability, ROC curves, confusion matrices and downloadable results |
| Decision tree | Final fitted tree diagram, retained features, training configuration and downloadable text rules |
| About the study | Dataset source, research method, model version and study limitations |

Enter values using the units shown, leaving unknown measurements blank. At least one measurement is required by the form. Select **Show prediction** to evaluate the submitted values. The prior result remains labelled as the last submission until you submit again. The result can be downloaded as JSON; patient entries are not saved into a patient database or used to retrain the model.

For a demonstration using the first dataset record: pregnancies **6**, glucose **148**, diastolic blood pressure **72**, skin thickness **35**, insulin **unknown**, BMI **33.6**, pedigree function **0.627**, age **50**. This is an example research record, not a clinical recommendation. The final model in the reference run produces a positive-class score of approximately **0.721**; a different retrained model may differ.

### Launch manually

From this project directory in PowerShell:

```powershell
.venv/Scripts/python.exe -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

Alternatively run `scripts/run_app.ps1`. Open the local URL printed by Streamlit. The interface includes patient inputs, a prediction with the actual tree path, research metrics, a tree diagram and dataset context. Enter measurements or leave fields unknown; the interface reports imputation and capping explicitly.

The local `.venv` and `.runtime/java` were prepared for this workspace. The final model must exist in `artifacts/final` before inference is available. If it has not yet been produced, run the evaluation and training commands below.

### Manual environment setup

Use Python 3.12 and a compatible 64-bit Java runtime (this project was tested with Java 17). Set `JAVA_HOME` to the Java installation. A project-local Java runtime at `.runtime/java/<distribution>/bin/server/jvm.dll` is detected automatically on Windows. See the [wrapper installation instructions](https://fracpete.github.io/python-weka-wrapper3/install.html).

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.lock
.venv/Scripts/python.exe -m pip install --no-deps -e .
```

On Linux/macOS use `.venv/bin/python` and system Java. Local environment folders and runtime downloads are excluded from Git. Dependencies are pinned in `requirements.lock`; the model manifest records exact Python, Java, WEKA and package versions. Only load trusted model artifacts: joblib and Java serialization are executable formats.

## Reproduce the research

The individual commands below are useful for development. For the complete checked and archived workflow, use **FullRun.bat**.

```powershell
# Recreate the uncleaned CSV from the preserved verified ARFF
.venv/Scripts/python.exe scripts/prepare_data.py

# Optional smaller integration run; outputs are marked smoke_only
.venv/Scripts/python.exe -m diabetes_c45.evaluate --smoke

# Full nested ten-fold outer / five-fold inner evaluation
.venv/Scripts/python.exe -m diabetes_c45.evaluate

# Retune on all records and save the final demonstration tree
.venv/Scripts/python.exe -m diabetes_c45.train

# Generate exportable figures and a factual results draft
.venv/Scripts/python.exe scripts/research_report.py

# Core and app integration checks (app checks require final artifacts)
.venv/Scripts/python.exe -m pytest -q
```

The full experiment is substantially slower than the smoke run. It prints progress at each outer fold and classifier. Configuration is in `config/experiment.json`. Do not interpret the smoke run as the dissertation result. Training requires the completed full evaluation with the matching data hash and configuration.

## Dataset and feature definitions

Use [OpenML dataset 37, version 1](https://www.openml.org/d/37), matching the proposal's Pima study design. It contains **768 records**, **8 predictors**, and original `Outcome` labels: **500 negative (0)** and **268 positive (1)**. The original root-level `diabetes.csv` and `diabetes1.csv` are identical copies of a different 403-record dataset and are not training or test inputs for this project.

| CSV field | Meaning / units |
|---|---|
| `Pregnancies` | Number of pregnancies; zero is valid |
| `Glucose` | Two-hour oral glucose tolerance test plasma glucose, mg/dL |
| `BloodPressure` | Diastolic pressure, mm Hg |
| `SkinThickness` | Triceps skinfold thickness, mm |
| `Insulin` | Two-hour serum insulin, micro-units/mL |
| `BMI` | Body mass index, kg/m² |
| `DiabetesPedigreeFunction` | The dataset's diabetes pedigree function |
| `Age` | Whole years, at least 21 for this study population |
| `Outcome` | Existing classification label; never a prediction input |

The five zero-as-missing fields contain: glucose **5**, blood pressure **35**, skin thickness **227**, insulin **374**, and BMI **11** zero entries. The CSV remains uncleaned on disk; fold-specific preprocessing handles these during training. See [data/README.md](data/README.md) for the source citation, column mapping and checksums. Git attributes preserve ARFF and derived CSV bytes across Windows checkouts so newline conversion does not invalidate their hashes.

## Experimental design

- The positive class is original Outcome 1, with a fixed score threshold of 0.5. Exactly tied scores are classified positive by this application, explicitly rather than depending on WEKA's class-index tie-break.
- Ten stratified outer folds estimate generalisation; five inner folds tune each model independently. All learned transforms fit only on their current training partition.
- Zero glucose, diastolic pressure, skin thickness, insulin and BMI are treated as missing. Numeric medians never use class labels. Zero pregnancies is retained.
- Candidate preprocessing: keep outliers or cap at training-derived 1.5-IQR bounds; retain eight predictors or select five using training-only mutual information with a fixed seed; no resampling or SMOTE.
- SMOTE neighbours are determined in training-standardised coordinates, then synthetic values are returned to original units. Fractional synthetic pregnancy/age values are retained during fitting and documented as a resampling limitation; real patient counts must be whole numbers.
- J48 searches confidence factors 0.05, 0.10, 0.25 and 0.40, and minimum leaf sizes 2, 5, 10 and 20. GaussianNB searches smoothing 1e-9, 1e-8 and 1e-7; scaled Logistic Regression searches C 0.1, 1 and 10.
- Inner mean ROC-AUC selects configurations; exact ties favour a smaller J48 tree and then stable search order. The two baselines use the same folds and preprocessing candidate sets.
- Six fixed J48 comparisons isolate median imputation, pruning, SMOTE, capping and feature selection. These comparisons share outer folds but are not tuned like the main three models.
- Every eligible record receives one held-out score per classifier. Reports include pooled metrics and separate fold means/standard deviations. Fold standard deviations are not confidence intervals or significance tests.
- Saved `row_id` values and split indices are zero-based positions in the verified CSV, excluding its header.
- The final J48 is retuned and fitted on all records for the demonstration. Its in-sample accuracy is not used as the generalisation estimate.

## Outputs

| Location | Contents |
|---|---|
| `data/README.md` | Provenance, schema, units, source checksum and label conversion |
| `results/full/` | Fold assignments, search logs, held-out scores, metrics, data audit, ROC curves and confusion matrices |
| `artifacts/final/` | J48 model and header, Python preprocessing, manifest, exact tree JSON, text rules and DOT diagram |
| `src/diabetes_c45/` | Validation, preprocessing, J48 adapter, nested evaluation, training and shared inference |
| `app/streamlit_app.py` | Streamlit prototype |
| `tests/` | Data contract, leakage boundaries, model reload, decision path and app checks |

Generated model artifacts are local and excluded from Git; research results and reproducible code can be tracked. No account system, patient database or cloud service is required.

```text
start.bat                       # Validate and demonstrate a saved model
FullRun.bat                     # Full research rebuild, tests and frontend
README.md
pyproject.toml
requirements.lock               # Pinned Python package versions
config/experiment.json          # Seeds, folds, parameter and preprocessing search
data/raw/                       # Preserved ARFF and OpenML metadata
data/derived/pima_diabetes.csv   # Reproducible, uncleaned Python training CSV
src/diabetes_c45/               # Python pipeline and WEKA adapter
app/streamlit_app.py            # Prototype interface
scripts/windows_runner.ps1     # Shared Windows workflow and failure handling
scripts/check_environment.py   # Runtime, dependency, data and artifact checks
scripts/prepare_data.py         # Checksum-verified conversion
scripts/research_report.py      # Figures and results draft
tests/                         # Core and application integration checks
results/full/                  # Completed evaluation outputs
artifacts/final/               # Saved model and associated preprocessing
docs/                          # Report, guide and optional browser screenshots
.runtime/logs/                 # Launcher transcripts (ignored by Git)
.runtime/backups/              # Prior generated outputs (ignored by Git)
.venv/                         # Local Python environment (ignored by Git)
```

## Troubleshooting

| Symptom | Action |
|---|---|
| Missing model or "Model preparation" page | Run `FullRun.bat`; a fresh checkout does not include ignored model artifacts. |
| Missing/wrong Python | Install 64-bit Python 3.12 and check `py -3.12 --version`. Rename an incompatible `.venv` before rebuilding. |
| Java cannot start | Check `JAVA_HOME` and Python/Java architecture. Use the local Java 17 runtime or rerun FullRun with a working internet connection when Java is missing. |
| Missing or mismatched packages | Run FullRun, or use the manual locked installation commands before `start.bat -CheckOnly`. Start never silently installs packages. |
| Dependency or Java download fails | Check connectivity and proxy settings, then rerun. The log contains the failed URL or package command. |
| Source checksum mismatch | Restore the original ARFF and metadata from the project/source. Do not bypass the check or substitute either original root CSV. |
| Model/evaluation hash mismatch | Run FullRun or restore a complete matching backup. Do not combine artifacts from different runs. |
| Old dashboard after retraining | Stop/restart the old Streamlit process and use the address printed by the new launcher. |
| Default port busy | Read the actual printed port, or use `-Port 8601`. Ten occupied ports cause a clear error. |
| Browser does not open | Open the printed local address manually. `-NoBrowser` intentionally disables automatic opening. |
| Long period of model fitting | Check the fold/classifier progress in the console or transcript. Full nested tuning is much slower than a single training fit. |
| Failed/aborted full run | Correct the first logged failure and rerun FullRun. It does not launch the frontend with incomplete new artifacts. |
| Native ARPACK warnings | WEKA can report unavailable optional native implementations while using its fallback. JVM preflight, successful inference and tests determine whether the workflow can continue. |

Do not share the prototype on a public interface as a clinical service. The provided launchers bind to `127.0.0.1` for local demonstrations.

## Interpretation and scope

The population consists of women aged at least 21 of Pima Indian heritage. Inputs must follow the original study definitions; glucose and insulin are two-hour measurements and blood pressure is diastolic. Scores represent a tree estimate for the existing dataset label, not a validated clinical probability or a specified future time-to-diabetes prediction. External and prospective validation are beyond this prototype.

Explanations follow J48's actual split objects rather than rounded text rules. Artifact hashes are verified when loading. The JVM starts once per process and shared inference uses a lock. Pytest's faulthandler is disabled because it interferes with the JVM's native handlers, as documented in the [JPype guide](https://jpype.readthedocs.io/en/latest/userguide.html#errors-reported-by-python-fault-handler).
