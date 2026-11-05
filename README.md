

# ICU Mortality Prediction Using Deep Learning
 
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Preprocessing-F7931E?logo=scikitlearn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-Statistics-8CAAE6?logo=scipy&logoColor=white)
![Colab](https://img.shields.io/badge/Run%20on-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
 
A data preparation and analysis pipeline for predicting **in-hospital mortality of Intensive Care Unit (ICU) patients** from vital signs, laboratory results, ventilation data and clinical severity scores. The project covers missing-data handling, outlier treatment, clinically guided EDA and statistical testing, preparing the data for a TensorFlow/Keras deep learning classifier.
 
---
 
## Table of Contents
 
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Key Findings](#key-findings)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [Author](#author)
---
 
## Problem Statement
 
ICU clinicians must constantly judge which patients are deteriorating and need escalated care. Early, data-driven mortality risk estimates can help:
 
- prioritise attention and resources for the highest-risk patients
- support conversations with families about prognosis
- benchmark ICU performance against expected outcomes
The task is **binary classification**:
 
| Label | Meaning | Patients |
|---|---|---|
| `0` | Survived | 3,097 (86.0%) |
| `1` | Died in hospital | 503 (14.0%) |
 
> ⚖️ The classes are **imbalanced** (about 1 death for every 6 survivors), so recall and precision on the mortality class matter more than raw accuracy.
 
---
 
## Dataset
 
**Source:** [Death Classification ICU — Kaggle](https://www.kaggle.com/datasets/fdemoribajolin/death-classification-icu)
 
The data is derived from ICU stay records (in the style of the PhysioNet/CinC Challenge 2012). It contains summary statistics for each patient's monitoring data.
 
- **Patients:** 3,600
- **Original columns:** 121 (features in `X_train_2025.csv`, target in `y_train_2025.csv`)
<details>
<summary><b>Feature groups (click to expand)</b></summary>
| Group | Examples |
|---|---|
| Severity scores | `SAPS-I`, `SOFA` |
| Demographics | `Age`, `Gender`, `Weight`, `Height` |
| ICU unit type | `CCU` (coronary care), `CSRU` (cardiac surgery recovery), `SICU` (surgical ICU) |
| Vital signs | Heart rate, invasive and non-invasive blood pressure (systolic / diastolic / mean), temperature, respiratory rate |
| Neurological | Glasgow Coma Scale (`GCS`) |
| Laboratory | BUN, creatinine, glucose, HCO3, pH, PaO2, PaCO2, FiO2, WBC, HCT, platelets, Mg, Na, K, lactate, troponin, liver enzymes |
| Ventilation | `MechVentStartTime`, `MechVentDuration`, `MechVentLast8Hour` |
| Renal output | `UrineOutputSum` |
 
Most time-series variables are summarised as **`_first`, `_last`, `_lowest`, `_highest`, `_median`** values over the stay.
 
</details>
---
 
## Project Workflow
 
### 1. Data Loading
Merged the feature file and the target file into a single DataFrame.
 
### 2. Missing Value Treatment
- Dropped **29 columns with more than 40% missing values**, including height, respiratory rate, SaO2, liver enzymes, albumin, bilirubin, cholesterol, lactate and troponin.
- Imputed the remaining numerical features: **median** if skewed (|skew| > 0.5), otherwise **mean**.
- Imputed categorical features (`Gender`, ICU unit flags, `MechVentLast8Hour`) with the **mode**.
- Dropped the `recordid` identifier. No duplicates were found.
- Result: **3,600 rows × 90 features + target**, with zero missing values.
### 3. Outlier Treatment: IQR Capping
- IQR analysis found outliers in many features. The most affected were `UrineOutputSum` (757), `MAP_lowest` (645) and `MechVentStartTime` (525).
- **Capping was chosen over removal.** In an ICU setting, extreme values (very low blood pressure, very high glucose) usually describe the sickest patients, not data errors, and removing them would erase the very cases the model must learn to recognise.
### 4. Exploratory Data Analysis
- **Target analysis:** mortality distribution
- **Univariate:** histogram, box plot and density plot for key vitals and labs
- **Bivariate:** age, glucose, blood pressure, heart rate, oxygen and urine output vs. mortality
- **Correlation heatmap** of severity scores, vitals and target
- **Multivariate:** SOFA vs. SAPS-I coloured by outcome
- **ICU unit analysis:** mortality rate by unit type
### 5. Statistical Analysis
 
**Numerical features: Pearson, Spearman and ANOVA (top associations)**
 
| Feature | Pearson r | Spearman ρ | ANOVA F | Interpretation |
|---|---|---|---|---|
| `GCS_last` | **−0.346** | −0.325 | **490.6** | Lower consciousness → higher mortality |
| `GCS_median` | −0.274 | −0.258 | 292.6 | |
| `BUN_last` | +0.243 | +0.218 | 226.5 | Kidney dysfunction |
| `GCS_highest` | −0.234 | −0.245 | 207.9 | |
| `BUN_first` | +0.213 | +0.198 | 170.8 | |
| `HCO3_last` | −0.181 | −0.176 | 121.8 | Metabolic acidosis |
| `SOFA` | +0.175 | +0.153 | 114.2 | Organ failure score |
| `SAPS-I` | +0.174 | +0.180 | 111.9 | Acute physiology score |
| `UrineOutputSum` | +0.171 | +0.171 | 108.6 | |
| `MechVentDuration` | +0.167 | +0.179 | 103.5 | Prolonged ventilation |
 
Almost every numerical feature was significantly associated with mortality. `HCT_first` (haematocrit) was the clearest exception.
 
**Categorical features: Chi-square test**
 
| Feature | χ² | p-value | Significant? |
|---|---|---|---|
| `MechVentLast8Hour` | 97.08 | < 0.001 | ✅ |
| `CSRU` | 85.74 | < 0.001 | ✅ |
| `Gender` | 1.14 | 0.286 | ❌ |
| `SICU` | 1.07 | 0.301 | ❌ |
| `CCU` | 0.00 | 1.000 | ❌ |
 
**Normality:** Shapiro–Wilk showed that nearly all features are non-normal.
 
---
 
## Key Findings
 
- 🧠 **Neurological status is the strongest single signal.** A low Glasgow Coma Scale at the end of the observation window has the highest association with death.
- 🩸 **Kidney function** (BUN, creatinine) and **acid–base balance** (HCO3, pH) are strong predictors, consistent with multi-organ failure.
- 📈 **Established severity scores work.** Higher SAPS-I and SOFA scores track mortality, which confirms the data behaves as expected clinically.
- 👴 **Age matters.** Patients who died were on average **70.2 years** old vs. **63.5 years** for survivors.
- 🫀 **Cardiac surgery recovery patients fare much better.** CSRU mortality was **3.6%** vs. **16.8%** for all other patients, reflecting planned post-operative admissions.
- 🫁 Patients still **on a ventilator in the last 8 hours** had significantly higher mortality.
---
 
## Tech Stack
 
| Category | Tools |
|---|---|
| Language | Python |
| Data handling | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Statistics | SciPy (`pearsonr`, `spearmanr`, `f_oneway`, `chi2_contingency`, `shapiro`) |
| Preprocessing | scikit-learn (`StandardScaler`) |
| Deep learning | TensorFlow / Keras |
| Environment | Google Colab |
 
---
 
## How to Run
 
### Option 1: Google Colab (recommended)
1. Download the dataset archive from [Kaggle](https://www.kaggle.com/datasets/fdemoribajolin/death-classification-icu).
2. Open the notebook in [Google Colab](https://colab.research.google.com/).
3. Upload the zip file to `/content/` (the first cell unzips it into `X_train_2025.csv` and `y_train_2025.csv`).
4. Run all cells.
### Option 2: Run locally
```bash
git clone https://github.com/<your-username>/icu-mortality-prediction.git
cd icu-mortality-prediction
pip install -r requirements.txt
jupyter notebook
```
Place `X_train_2025.csv` and `y_train_2025.csv` in the project folder and update the file paths in the notebook.
 
---
 
## Project Structure
 
```
icu-mortality-prediction/
├── ICU_Mortality_Prediction.ipynb   # Cleaning, EDA and statistical analysis
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```
 
---
 
## Roadmap
 
- [x] Data merging and missing-value treatment
- [x] Outlier capping
- [x] Clinically guided EDA
- [x] Statistical testing (Pearson, Spearman, ANOVA, Chi-square, Shapiro–Wilk)
- [ ] Feature transformation and engineering (e.g. BUN/creatinine ratio, blood-pressure ranges)
- [ ] Stratified train/test split and scaling
- [ ] Deep neural network classifier (Keras) with class weighting and early stopping
- [ ] Evaluation with recall, precision, F1, ROC-AUC and confusion matrix
- [ ] Benchmark against the SAPS-I / SOFA scores alone and against gradient boosting
**Technical notes for the modelling stage**
- **Handle the `-1` placeholder in `SAPS-I`.** Missing SAPS-I scores are coded as `-1`, which capping turned into `0.5`. Converting `-1` to NaN before imputation would keep these patients from looking artificially healthy.
- **Missingness can be informative.** In ICU data, whether a test was ordered (lactate, troponin) often signals how concerned clinicians were. Adding binary "was measured" flags for the dropped columns can recover that signal.
- **Prevent leakage.** Imputation statistics and capping bounds are currently computed on the full dataset. Fitting them on the training split only, inside a scikit-learn `Pipeline`, keeps the test set truly unseen.
- **Reduce redundancy.** Many features are near-duplicates (`_first`/`_last`/`_median` of the same vital, and invasive vs. non-invasive blood pressure). Correlation-based pruning or regularisation (`Dropout`, L2) will help a network trained on only 3,600 patients.
---
 
## Disclaimer
 
This project is for **educational and research purposes only**. It is not a clinical decision-support tool and must not be used to make decisions about patient care.
 
---
 
## Author
 
**Patricia Fuglo**
 
⭐ If you found this project useful, consider giving it a star!
 
