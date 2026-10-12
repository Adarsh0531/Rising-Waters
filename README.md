# FloodWatch

FloodWatch is a Flask prototype that trains and compares Decision Tree, Random Forest, K-Nearest Neighbours, and XGBoost models from a labelled CSV.

## Dataset

Datasets are intentionally excluded from this repository by `.gitignore`. To run the default model, provide an authorized copy of the labelled Kerala rainfall CSV at `data\kerala.csv`.

The default CSV used for development contains yearly Kerala rainfall observations and a `FLOODS` yes/no label. The app trains a binary classifier and shows the estimated class and model probability. Its inputs are generated from the CSV columns; constant columns such as `SUBDIVISION` are removed automatically. You can also provide a CSV containing a continuous `FloodProbability` target to train the regression version.

The Kerala dataset has historical, annual records rather than live district weather feeds or cloud visibility. It is not included here; check its source's license and usage terms before obtaining or redistributing it. The classifier's development holdout score was 91.67% on 24 later-year records. This small historical evaluation is not proof of operational forecasting accuracy. Do not present it as a guaranteed early warning or use this prototype alone for evacuation or emergency resource decisions.

## Setup (Windows)

From the project folder, open PowerShell and run:

```powershell
New-Item -ItemType Directory -Force data
# Copy your authorized labelled file to data\kerala.csv before training.
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py
python app.py
```

Create a virtual environment and install dependencies if needed:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Open <http://127.0.0.1:5000>. The default command trains against `data\kerala.csv`. To choose a probability-regression CSV or another labelled CSV:

```powershell
python train.py --data data\flood.csv
python train.py --data path\to\your-labelled-data.csv --artifact-dir artifacts
```

Training automatically selects classification for `FLOODS` and regression for `FloodProbability`. If the dataset has a `YEAR` column, validation and holdout sets use later years than the training data to reduce temporal leakage; otherwise the script uses a seeded random split (stratified for classification). Classifiers are selected by validation accuracy; regressors by validation RMSE. The selected model is refit on the training and validation rows, then evaluated on the untouched holdout set. Metrics and the selected model are saved in `artifacts\`.

The optional `PORT` environment variable controls the Flask listening port for hosting platforms. Deployment to IBM Cloud and ingestion of current rainfall/weather measurements are not configured.
