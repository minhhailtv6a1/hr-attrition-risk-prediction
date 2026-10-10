# AGENTS.md - Operational Runbook & Architecture Guide (HR Attrition Risk Prediction & LLM Reasoning)

This document serves as an exhaustive, authoritative guide for **AI Coding Agents** and **MLOps/Data Engineers** when interacting with, maintaining, debugging, or extending the `hr_attrition_risk_prediction` project.

---

## 1. Project Overview & Business Problem

This system is an End-to-End Machine Learning and Explainable AI reasoning platform designed to solve a critical human resources challenge: **Predicting whether an employee will leave the organization within the next 90 days (`left_within_90_days`) and providing structured, guardrailed explanations and retention recommendations for HR managers**.

### Technology Stack & Frameworks

- **Programming Language:** Python 3.11+
- **Machine Learning:** Scikit-Learn, XGBoost, LightGBM, Imbalanced-learn techniques
- **MLOps & Experiment Tracking:** MLflow (Nested Runs, Parameter & Metric Tracking, Model Registry/Artifacts)
- **Serving & API:** FastAPI, Uvicorn, Pydantic v2, Pydantic Settings
- **LLM Reasoning & Guardrails:** OpenAI API (`gpt-4o-mini`) coupled with a deterministic heuristic fallback generator and People Analytics Guardrails
- **Data & Storage:** Parquet (PyArrow), Pandas, AWS S3 (Boto3), PostgreSQL
- **Containerization:** Docker (Multi-stage/specialized builds), Docker Compose

### 4 Core Technical Pillars

1. **Zero Temporal Data Leakage:**
   - Strict omission of any attributes occurring after the separation event (`termination_date`, `attrition_type`).
   - Implementation of point-in-time joins using `pd.merge_asof(direction="backward", by="employee_id")`, guaranteeing that at any given observation date (`snapshot_date`), the model only observes strictly historical data ($\le \text{snapshot\_date}$).
2. **Severe Class Imbalance Handling:**
   - Real-world attrition rate is heavily skewed (~**4.29%** positive rate across ~128,540 rows).
   - Applied Hybrid Resampling: Stratified Downsampling of the majority class (`label=0`) combined with Oversampling (with replacement) of the minority class (`label=1`) stratified across job level, department, and employment type.
3. **Quantitative 4-Tier Risk Band Stratification:**
   - Rather than outputting a raw binary classification label, calibrated probabilities (`churn_probability`) are categorized into 4 operational business risk bands: `CRITICAL` ($\ge 0.75$), `HIGH` ($0.50 - 0.74$), `MEDIUM` ($0.25 - 0.49$), and `LOW` ($< 0.25$).
4. **Explainability with Strict People Analytics Guardrails:**
   - Strictly prohibits hallucinating or speculating about private life factors outside observable data.
   - Non-deterministic assertions only (never claims an employee will definitely leave).
   - Enforces an explicit "Decision-Support System" disclaimer.
   - Generates 2-3 constructive, supportive 1-on-1 discussion topics for the direct manager.

---

## 2. Directory Structure & Module Responsibilities

```text
hr_attrition_risk_prediction/
├── AGENTS.md                             # Agent guidance and system manual (this file)
├── README.md                             # Human-facing project documentation
├── .env.example                          # Environment configuration template
├── .env                                  # Active environment config (ignored by git)
├── .gitignore                            # Git exclusion rules
├── pyproject.toml                        # Build system specifications and pytest config
├── requirements.txt                      # Full dependencies for training, MLflow, and data pipelines
├── requirements-api.txt                  # Minimal dependencies for the FastAPI serving container
├── Dockerfile                            # Training & pipeline execution container image
├── Dockerfile.api                        # Production FastAPI serving container image
├── docker-compose.yml                    # Multi-container orchestration: Postgres, MLflow, FastAPI
├── helper/
│   └── helper_functions.py               # Utility functions (safe JSON writing, Parquet loading)
├── app/
│   ├── __init__.py                       # Package version declaration (1.0.0)
│   ├── config.py                         # Pydantic Settings resolving .env with absolute paths
│   ├── features.py                       # Static definition of 32 modeling features (25 num, 7 cat)
│   ├── schemas.py                        # Pydantic models for API request/response validation
│   ├── data.py                           # ETL, S3 sync, temporal merge, validation, preprocessing
│   ├── train.py                          # Training pipeline, MLflow tracking, model serialization
│   ├── evaluate.py                       # Evaluation metrics (PR-AUC, ROC-AUC, F1) & risk band evaluation
│   ├── inference.py                      # Model loading, probability scoring, risk band/driver assignment
│   ├── explain.py                        # Risk driver extraction, LLM prompt guardrails, OpenAI API call
│   └── main.py                           # FastAPI application endpoints and lifecycle handler
├── data/
│   ├── raw/                              # Destination for raw CSV files (local or synced from S3)
│   ├── merged/                           # Parquet file after anti-leakage temporal join (merged_data.parquet)
│   ├── processed/                        # Preprocessed, engineered, and feature-selected Parquet datasets
│   └── samples/                          # Sample JSON payloads for CLI testing and validation
├── models/
│   ├── best_model.joblib                 # Serialized pipeline of the top-performing model
│   ├── metrics.json                      # Comprehensive performance summary across all candidate models
│   ├── schema.json                       # Feature metadata schema consumed by /model-info
│   └── production/
│       └── model_manifest.json           # Model release metadata, timestamps, and quality gate audit
├── reports/
│   ├── eda_charts/                       # Correlation and feature distribution charts (.png)
│   └── evaluations/                      # Confusion matrices and PR-curves per risk band
├── scripts/
│   ├── prepare_data.py                   # CLI: End-to-end data ingestion, validation, and preprocessing
│   ├── eda.py                            # CLI: Automated exploratory data analysis chart generator
│   ├── train_model.py                    # CLI: Model training, MLflow logging, and artifact generation
│   ├── evaluate_model.py                 # CLI: Detailed model evaluation on the test set
│   └── predict_sample.py                 # CLI: Single-sample prediction from a JSON file
├── notebooks/
│   └── 01_hr_attrition_end_to_end.ipynb  # End-to-end demonstration notebook
└── tests/
    ├── __init__.py
    ├── test_leakage.py                   # Automated tests for temporal data leakage prevention
    ├── test_schemas.py                   # Automated tests for Pydantic schema validation
    └── test_inference.py                 # Automated tests for inference logic and risk band assignment
```

---

## 3. Data Architecture & Anti-Leakage Protocol

### 6 Source CSV Tables (`data/raw/`)

1. `attrition_labels.csv`: Primary target labels (`left_within_90_days` $\rightarrow$ `label`), `snapshot_date`, `employee_id`. _(Mandatory: Immediately drop `termination_date` and `attrition_type`)_.
2. `employees.csv`: Static/master employee data (`hire_date`, `department`, `job_level`, `role`, `location`, `salary_band_min_usd`, `salary_band_max_usd`, `monthly_salary_usd`, `promotion_count`, `last_promotion_date`).
3. `engagement_surveys.csv`: Periodic employee sentiment surveys (`survey_date`, `engagement_score`, `work_life_balance_score`, `manager_support_score`, `career_growth_score`, `compensation_satisfaction_score`).
4. `leave_logs.csv`: Historical leave logs (`leave_date`, `leave_type`, `leave_days`, `is_unplanned`, `approved_by_manager`).
5. `performance_reviews.csv`: Performance evaluations (`review_date`, `performance_score`, `goal_completion_rate`, `manager_rating`, `peer_feedback_score`, `review_sentiment_score`).
6. `work_activity.csv`: Weekly operational telemetry (`week_start`, `tasks_completed`, `late_tasks`, `overtime_hours`, `meeting_hours`, `focus_hours`, `after_hours_messages`, `absent_days`).

### Anti-Leakage Merging Workflow (`app/data.py: _merge_temporal_data`)

```text
attrition_labels ──[merge on employee_id]──> + employees (static attributes)
        │
        ├──[merge_asof: snapshot_date >= survey_date]────> + engagement_surveys
        ├──[merge_asof: snapshot_date >= leave_date]─────> + leave_logs
        ├──[merge_asof: snapshot_date >= review_date]────> + performance_reviews
        └──[merge_asof: snapshot_date >= week_start]─────> + work_activity
                                                                │
                                                                ▼
                                                   merged_data.parquet (128,540 rows)
```

**Mandatory Merge Rules:**

- Both primary and secondary DataFrames must be sorted by `[date_col, "employee_id"]` before executing `pd.merge_asof`.
- Always set `direction="backward"` to guarantee that only the most recent historical observation relative to `snapshot_date` is retrieved.

---

## 4. Feature Engineering & Class Imbalance Strategy

### 32 Production Modeling Features (`app/features.py`)

Models must train strictly on these **32 features** (excluding `employee_id` and the target `label`):

- **25 Numeric Features (`NUMERIC_FEATURES`):**
  `tasks_completed`, `late_tasks`, `overtime_hours`, `meeting_hours`, `focus_hours`, `after_hours_messages`, `absent_days`, `performance_score`, `goal_completion_rate`, `manager_rating`, `peer_feedback_score`, `review_sentiment_score`, `leave_days`, `manager_support_score`, `career_growth_score`, `compensation_satisfaction_score`, `job_level`, `salary_band_min_usd`, `salary_band_max_usd`, `monthly_salary_usd`, `promotion_count`, `engagement_score`, `work_life_balance_score`, `tenure_months` _(derived: snapshot_date - hire_date)_, `days_since_last_promotion` _(derived: snapshot_date - last_promotion_date)_.
- **7 Categorical Features (`CATEGORICAL_FEATURES`):**
  `leave_type`, `is_unplanned`, `approved_by_manager`, `department`, `role`, `location`, `employment_type`.

### Class Balancing Strategy (`handle_imbalance_data`)

- **Training Set:**
  - Stratified downsampling of the majority class (`label=0`) to `33,000` samples using `DOWN_SAMPLE_STRATIFY_FEATURES`.
  - Upsampling with replacement of the minority class (`label=1`) to `16,500` samples.
  - Final post-balance training ratio: **2:1** (49,500 total rows).
- **Test Set:**
  - Majority class downsampled to `2,300` samples to accelerate evaluation while retaining 100% of natural minority test occurrences for unbiased PR-AUC and ROC-AUC benchmarking.

---

## 5. Model Training & MLflow Tracking (`app/train.py`)

### 1. Preprocessor Pipeline (`build_preprocessor`)

Constructed using standard scikit-learn transformers:

- **Numeric Branch:** `SimpleImputer(strategy="mean")` $\rightarrow$ `StandardScaler()`
- **Categorical Branch:** `SimpleImputer(strategy="most_frequent")` $\rightarrow$ `OneHotEncoder(handle_unknown="ignore")`

### 2. Candidate Model Pipelines

Five baseline and ensemble classifiers are trained concurrently with hyperparameters tuned for imbalanced data:

1. **DecisionTree:** `class_weight="balanced"`, `max_depth=5`
2. **RandomForest:** `n_estimators=400`, `class_weight="balanced_subsample"`, `n_jobs=-1`
3. **GradientBoosting:** `n_estimators=400`, `random_state=42`
4. **XGBoost:** `n_estimators=400`, `scale_pos_weight=2.0`, `eval_metric="logloss"`
5. **LightGBM:** `n_estimators=400`, `class_weight="balanced"`, `verbose=-1`

### 3. MLflow Tracking & Nested Runs

- Every training cycle executes under a **Parent Run** (`model_training_batch`).
- Each individual model executes within a nested **Child Run** matching its algorithm name.
- Tracks `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `pr_auc`, and risk band distributions.
- Top model selected by highest test `roc_auc` and exported to `models/best_model.joblib`.
- Quality Gate validation: Verifies `roc_auc >= MIN_ROC_AUC` (default `0.70`) and records the status in `models/production/model_manifest.json`.

---

## 6. Risk Stratification & LLM Guardrails

### Risk Band Classification Logic (`app/evaluate.py: assign_risk_band`)

| Attrition Probability (`churn_probability`) |   Risk Band    | Business Meaning                         | Recommended HR Intervention                                                                             |
| :------------------------------------------ | :------------: | :--------------------------------------- | :------------------------------------------------------------------------------------------------------ |
| $\ge 0.75$                                  | **`CRITICAL`** | Imminent, high-urgency departure risk    | Immediate 1-on-1 intervention within 48h; review compensation adjustments or project transfers          |
| $0.50 - 0.74$                               |   **`HIGH`**   | High risk, clear dissatisfaction signals | Direct manager schedules an in-depth conversation; review workload and promotion timeline within 7 days |
| $0.25 - 0.49$                               |  **`MEDIUM`**  | Moderate risk, emerging retention flags  | Conduct periodic satisfaction check-ins; gather training and work-life balance feedback                 |
| $< 0.25$                                    |   **`LOW`**    | Stable, well-engaged                     | Maintain current growth roadmap; provide regular milestone recognition                                  |

### Strict LLM Guardrails for HR Analytics (`app/explain.py`)

When generating management explanations via `/explain`, the system enforces 5 non-negotiable prompt guardrails:

1. **No Hallucination/Speculation:** Never extrapolate or speculate on personal, family, or psychological factors not present in the top 3 observable risk drivers (`top_drivers`).
2. **Probabilistic Tone (Non-Judgmental):** Always use probabilistic phrasing ("exhibits signs of", "indicates potential risk"). Never claim that an employee "will definitely resign".
3. **Decision-Support Disclaimer:** Must include a clear disclaimer stating that this AI output serves as decision-support guidance only and does not replace human management judgment.
4. **Constructive Actions:** Propose 2-3 supportive, constructive conversational topics rather than confrontational questions.
5. **Offline Fallback:** If `OPENAI_API_KEY` is not provided or network requests fail, the service must automatically fallback to a deterministic, rule-based template that satisfies 100% of these safety criteria.

---

## 7. FastAPI Endpoints Reference

Default server runs on port `8080`:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

| Method | Endpoint         | Purpose                                       | Main Input                                 | Main Output                                                                      |
| :----- | :--------------- | :-------------------------------------------- | :----------------------------------------- | :------------------------------------------------------------------------------- |
| `GET`  | `/health`        | Check service & model health                  | None                                       | `status: "healthy"/"degraded"`, `model_loaded: bool`                             |
| `GET`  | `/model-info`    | Inspect model schema metadata & quality gate  | None                                       | `model_name`, `numeric_features`, `categorical_features`, `quality_gate_roc_auc` |
| `GET`  | `/model-metrics` | Retrieve evaluation metrics of trained models | None                                       | `best_model`, `full_metrics` (Accuracy, F1, ROC-AUC, PR-AUC)                     |
| `POST` | `/predict`       | Predict risk for a single employee            | `PredictRequest` (32 features)             | `churn_probability`, `risk_band`, `top_drivers`, `recommended_action`            |
| `POST` | `/batch-predict` | Predict risk for a batch of employees         | `BatchPredictRequest` (array of employees) | Ranked list of predictions sorted by probability descending                      |
| `POST` | `/explain`       | Generate LLM-powered HR retention reasoning   | `ExplainRequest` (drivers, score)          | `explanation_text`, `suggested_topics`, `guardrail_status: "PASSED"`             |
| `POST` | `/analysis-runs` | Persist batch prediction results to database  | `AnalysisRunsRequest`                      | `analysis_run_id`, `saved_to_database`, `total_processed`                        |

---

## 8. Environment Variables Specification (`.env`)

| Variable                 |  Type  |              Default              | Description & Recommended Settings                                     |
| :----------------------- | :----: | :-------------------------------: | :--------------------------------------------------------------------- |
| `DATA_SOURCE`            | string |              `local`              | Data origin: `local` (read from `data/raw`) or `s3` (sync from AWS S3) |
| `ARTIFACTS_DIR`          |  path  |            `artifacts`            | Directory storing JSON pipeline reports                                |
| `RAW_DATA_DIR`           |  path  |            `data/raw`             | Directory storing the 6 source CSV files                               |
| `MERGED_DATA_DIR`        |  path  |           `data/merged`           | Directory storing the anti-leakage merged Parquet file                 |
| `PROCESSED_DATA_DIR`     |  path  |         `data/processed`          | Directory storing preprocessed and split Parquet files                 |
| `MODELS_DIR`             |  path  |             `models`              | Directory storing model binaries and manifests                         |
| `REPORTS_DIR`            |  path  |             `reports`             | Directory storing evaluation reports and EDA charts                    |
| `S3_BUCKET`              | string |               `""`                | AWS S3 Bucket containing source data (required if `DATA_SOURCE=s3`)    |
| `S3_PREFIX`              | string |              `data/`              | Prefix path within the S3 bucket                                       |
| `AWS_DEFAULT_REGION`     | string |         `ap-southeast-1`          | AWS Region                                                             |
| `AWS_ACCESS_KEY_ID`      | string |               `""`                | AWS IAM Access Key ID                                                  |
| `AWS_SECRET_ACCESS_KEY`  | string |               `""`                | AWS IAM Secret Access Key                                              |
| `MLFLOW_TRACKING_URI`    | string |      `http://localhost:5001`      | Address of the MLflow Tracking Server                                  |
| `MLFLOW_EXPERIMENT_NAME` | string | `hr-attrition-risk-prediction-v1` | Name of the MLflow Experiment                                          |
| `MIN_ROC_AUC`            | float  |              `0.70`               | Minimum ROC-AUC threshold required to pass the release quality gate    |
| `DATABASE_URL`           | string |               `""`                | PostgreSQL connection string (e.g. `postgresql+psycopg2://...`)        |
| `OPENAI_API_KEY`         | string |               `""`                | OpenAI API key (optional; if empty, activates offline fallback)        |
| `LLM_MODEL`              | string |           `gpt-4o-mini`           | OpenAI model ID used for HR reasoning                                  |

---

## 9. Step-by-Step Agent Runbook

When an Agent is tasked with executing or validating the project pipeline, **always adhere strictly to the following execution sequence**:

### Step 1: Environment & Dependency Initialization

```powershell
# Activate virtual environment (PowerShell on Windows)
.\myenv\Scripts\Activate.ps1

# Install all project dependencies
pip install -r requirements.txt
```

### Step 2: End-to-End Data Preparation

Executes sequentially: S3 Sync (if enabled) $\rightarrow$ Validate $\rightarrow$ Preprocess $\rightarrow$ Feature Engineering $\rightarrow$ Feature Selection.

```bash
python scripts/prepare_data.py
```

_Expected Output:_ Generates `data/processed/feature_selected.parquet` containing 34 columns ready for training.

### Step 3: Generate Exploratory Data Analysis Charts

```bash
python scripts/eda.py
```

_Expected Output:_ Generates 6 bivariate PNG charts in `artifacts/eda_charts/` and `artifacts/eda_summary.json`.

### Step 4: Model Training & Experiment Tracking

> **Important:** Ensure the MLflow Tracking Server is accessible at the configured tracking URI before running. The default URI is `http://localhost:5001`; launch the matching local server with `mlflow server --host 127.0.0.1 --port 5001`, or start the Compose services with `docker compose up -d postgres mlflow`.

```bash
python scripts/train_model.py
```

_Expected Output:_

- Generates `models/best_model.joblib`
- Generates `models/metrics.json`
- Generates `models/production/model_manifest.json`

### Step 5: Run Automated Unit Tests

```bash
pytest tests/ -v
```

### Step 6: Launch FastAPI Serving Service

**Option 1: Direct Local Execution**

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8080 --reload
```

Health Check: `curl http://127.0.0.1:8080/health`

**Option 2: Containerized Execution with Docker Compose**

```bash
docker compose up --build -d
```

Starts all supporting services concurrently:

- PostgreSQL Database: port `5433`
- MLflow Tracking UI: `http://localhost:5001`
- FastAPI Prediction Service: `http://localhost:8080` (API Docs at `/docs`)

---

## 10. Known Quirks, Traps & Maintenance Rules

When modifying or refactoring codebase logic, Agents must adhere strictly to these engineering constraints:

### 1. Working Directory (CWD) Path Resolution (`app/config.py`)

- When running code from subdirectories (e.g. from `notebooks/` in Jupyter or from `scripts/`), `Path.cwd()` points to the child folder rather than the project root.
- **Rule:** In `app/config.py`, always anchor all directory paths relative to `BASE_DIR = Path(__file__).resolve().parent.parent`. Never construct relative path instances without chaining through `BASE_DIR`.

### 2. Type Mismatch: `bool` vs `NoneType` in Parquet

- When merging datasets and writing to Parquet, missing values in boolean columns (e.g. `is_unplanned`, `approved_by_manager`) are saved by PyArrow as Python **`None` (`NoneType`)**, rather than NumPy's in-memory `np.nan` (float).
- Calling `pd.cut` or applying inequality operators on such a column causes Python 3 to fail:
  `TypeError: '<' not supported between instances of 'bool' and 'NoneType'`
- **Remedy:** Always coerce boolean columns with nulls to numeric representations before applying binning or range cuts:
  ```python
  unplanned_numeric = (
      frame["is_unplanned"]
      .replace({True: 1, False: 0, "True": 1, "False": 0})
  )
  unplanned_numeric = pd.to_numeric(unplanned_numeric, errors="coerce").fillna(0)
  ```

### 3. Collection Types: `SOURCE_FILES` as List vs Set

- In `app/config.py` and `app/data.py`, `SOURCE_FILES` is defined as a `list`.
- **Rule:** Never invoke `SOURCE_FILES.difference(...)` directly (lists do not have a `difference` method). Always cast to `set(SOURCE_FILES).difference(...)`.

### 4. Mandatory Generation of `models/schema.json`

- Endpoint `GET /model-info` in `app/main.py` reads directly from `settings.SCHEMA_PATH` (`models/schema.json`).
- When modifying `train_and_track_models` in `app/train.py`, always ensure that `models/schema.json` is exported with feature names and model metadata to prevent 404 errors on this endpoint.

### 5. Essential Container Dependencies (`psycopg2-binary` & `httpx`)

- When using PostgreSQL as the MLflow backend store in Docker Compose, `psycopg2-binary` is required inside the image.
- In `requirements-api.txt`, ensure that `httpx` is included so that OpenAI API calls initiated by `app/explain.py` can execute outbound HTTP requests from inside the container.
