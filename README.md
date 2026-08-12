[README.md](https://github.com/user-attachments/files/30851902/README.md)
# 🩺 Diabetes Risk Screening — Logistic Regression

**Live demo:** [https://diabetese-screening.streamlit.app/](https://diabetese-screening.streamlit.app/)

A logistic regression model that estimates diabetes risk from routine diagnostic
measurements, built as an interpretable screening tool rather than a black-box
classifier. The project emphasizes *why* a prediction is made — odds ratios,
statistical significance, and a deliberately tuned decision threshold — over
squeezing out marginal accuracy gains.

---

## Objective

To build an interpretable logistic regression model that predicts the likelihood
of diabetes onset in patients based on diagnostic measurements (glucose level,
BMI, blood pressure, age, etc.), with emphasis on identifying key risk factors,
quantifying their effect via odds ratios, and selecting a decision threshold
appropriate for a low-cost screening context — where the cost of missing a
diabetic patient (false negative) is weighed against the cost of unnecessary
follow-up testing (false positive).

---

## Dataset

**Pima Indians Diabetes Dataset** — 768 rows, 8 diagnostic features
(`Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`,
`DiabetesPedigreeFunction`, `Age`), binary target `Outcome`.

Class balance: **65.1% non-diabetic / 34.9% diabetic** (mild-to-moderate
imbalance — not extreme enough to require resampling, but enough that raw
accuracy is a misleading metric on its own).

---

## Workflow

1. **EDA** — checked class balance, feature distributions by outcome, and a
   correlation heatmap.
2. **Missing value handling** — several columns contain biologically
   impossible zeros (a blood pressure or BMI of 0 doesn't exist), which were
   treated as missing rather than real measurements.
3. **Missingness flagging** — for the two heavily-affected columns
   (`SkinThickness`, `Insulin`), added binary indicator features
   (`SkinThickness_missing`, `Insulin_missing`) *before* imputing, so the model
   can separately learn from "was this ever measured" as its own signal.
4. **Train/test split** — 80/20, stratified on the target to preserve class
   balance in both sets.
5. **Imputation** — median imputation, fit on the training set only and
   applied to both train and test, to avoid data leakage.
6. **Scaling** — continuous features standardized (`StandardScaler`); the
   binary missingness flags were left unscaled to keep them interpretable.
7. **Modeling** — logistic regression, coefficients converted to odds ratios.
8. **Statistical validation** — significance testing (`statsmodels`) and
   multicollinearity check (VIF) on top of the sklearn model.
9. **Threshold tuning** — precision/recall tradeoff analyzed across
   thresholds and a final operating threshold selected deliberately (see
   below), instead of using sklearn's default 0.5.
10. **Deployment** — Streamlit app serving real-time predictions using the
    saved model, imputer, and scaler.

---

## Key decisions and why

**Zeros → missing, not real values.**
Several features had a small number of zeros (`Glucose`: 5, `BloodPressure`:
35, `BMI`: 11) that are simple data-entry gaps, but two features had much
higher rates (`SkinThickness`: 227 rows / ~30%, `Insulin`: 374 rows / ~49%).
Given the scale of missingness on those two, they were treated as a
potentially systematic gap (patients who weren't given the more invasive
test) rather than random noise — a **Missing Not At Random (MNAR)** hypothesis.

**Flag-and-impute over drop or plain impute.**
Rather than dropping `SkinThickness`/`Insulin` entirely or silently filling
them, both a missingness flag *and* an imputed value were kept. This lets the
model use "was this test done" as its own feature, separate from the
(possibly unreliable) imputed value itself. The missingness rate was checked
against `Outcome` and found modestly higher among diabetics (+5pp
`SkinThickness`, +4pp `Insulin`), a direction consistent with the MNAR
hypothesis — though after fitting the full model with `statsmodels`, neither
missingness flag reached conventional statistical significance (p = 0.565 and
p = 0.152 respectively). The coefficient direction is suggestive, not
conclusive, and is reported as such rather than overstated.

**Leakage discipline.**
Every step that computes a statistic from the data (imputation medians,
scaling parameters) was fit on the training set only and applied to the test
set via `.transform()`, never `.fit_transform()` on test data.

**Threshold tuning over class weighting.**
Given the mild imbalance (65/35, not severe) and the project's stated
screening-context goal, class weighting was considered but not used — it
would distort the model's predicted probabilities, undermining the goal of
interpretable, calibrated risk estimates. Instead, the default 0.5 threshold
was replaced with a threshold chosen directly from the precision-recall
tradeoff, which keeps the model's probability estimates untouched and lets
the *decision rule* reflect the real-world cost asymmetry (a missed diabetic
diagnosis is more costly than an unnecessary follow-up test).

**Final threshold: 0.25**
At the default 0.5 threshold, recall on the diabetic class was only 52% —
missing nearly half of actual diabetics. Lowering the threshold to **0.25**
raised recall to **89%** (48 of 54 diabetics caught in the test set, only 6
missed) while precision only dropped slightly (0.57 → 0.58, i.e. essentially
unchanged). This was chosen because, past this point, further lowering the
threshold offered no additional recall gain but did cost precision — 0.25 was
the most favorable point on the tradeoff curve for this project's stated
priorities.

---

## Results

| Metric | Threshold 0.5 (default) | Threshold 0.25 (chosen) |
|---|---|---|
| Accuracy | 0.69 | 0.73 |
| Precision (diabetic) | 0.57 | 0.58 |
| Recall (diabetic) | 0.52 | 0.89 |
| ROC-AUC | 0.815 | 0.815 (threshold-independent) |

**Top features by odds ratio** (effect per 1 standard deviation increase,
scaled features):

| Feature | Odds Ratio | Statistically significant? |
|---|---|---|
| Glucose | 3.31x | Yes (p < 0.001) |
| BMI | 2.01x | Yes (p < 0.001) |
| Pregnancies | 1.45x | Yes (p = 0.002) |
| DiabetesPedigreeFunction | 1.28x | Yes (p = 0.024) |
| Insulin_missing | 1.47x | No (p = 0.152) |
| Age | 1.14x | No (p = 0.317) |
| SkinThickness | 1.03x | No (p = 0.855) |
| Insulin | 0.96x | No (p = 0.719) |
| BloodPressure | 0.94x | No (p = 0.546) |
| SkinThickness_missing | 0.87x | No (p = 0.565) |

All VIF values were below 3, ruling out meaningful multicollinearity across
all 10 features.

---

## Repository structure

```
.
├── app.py                  # Streamlit app
├── requirements.txt        # Dependencies
├── artifacts/
├── imputer.pkl          # Fitted SimpleImputer (train-only medians)
├── scaler.pkl           # Fitted StandardScaler (train-only stats)
├── logistic_regression_model.pkl
└── config.json          # Threshold + feature order
└── notebooks/               # EDA, modeling, and evaluation notebooks
```

---

## Running locally

```bash
# Clone the repo
git clone https://github.com/nayeem29dse/Diabetese-Detection-with-streamlit-deployment.git
cd Diabetese-Detection-with-streamlit-deployment

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`. Make sure the `artifacts/`
folder (containing the saved imputer, scaler, and model) is present in the
same directory as `app.py`.

---

## Limitations and possible improvements

- **Sample size** — 768 rows is small; several coefficients (Age,
  BloodPressure, the missingness flags) didn't reach statistical significance,
  possibly due to limited power rather than a true null effect. A larger
  dataset could clarify these.
- **Imputation strategy** — median imputation was used uniformly for
  simplicity. A documented comparison against KNN imputation (especially for
  the heavily-missing `SkinThickness`/`Insulin` columns) would be a natural
  next step.
- **Calibration** — the model's predicted probabilities have not yet been
  formally checked with a calibration curve. Since the app surfaces raw
  probability estimates to the user, this would strengthen the claim that a
  "70% probability" genuinely reflects a ~70% real-world rate.
- **Population generalizability** — the Pima Indians Diabetes dataset reflects
  one specific population; performance on a more diverse population is
  untested.
- **Single-model comparison** — only logistic regression was fit, by design
  (this project's goal was interpretability). A brief benchmark against a
  tree-based model (e.g. LightGBM) would help quantify the accuracy/
  interpretability tradeoff explicitly.

  ## Author

**Mehedee Hasan Nyeem**

**Data Science Student**

---

## Disclaimer

This tool is for educational and portfolio purposes only. It is not a medical
device and should not be used as a substitute for professional diagnosis.
