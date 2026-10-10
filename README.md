# HR Attrition Risk Prediction & LLM Reasoning API

Dự án Machine Learning & MLOps hoàn chỉnh cho bài toán **Dự đoán rủi ro nhân viên nghỉ việc trong 90 ngày (`left_within_90_days`) kết hợp LLM Reasoning có Guardrails** để hỗ trợ phòng Nhân sự (HR/People Analytics) ra quyết định can thiệp sớm và có trách nhiệm.

---

## Bối cảnh nghiệp vụ

Trong môi trường doanh nghiệp tăng trưởng nhanh, đội ngũ HR và People Analytics phải theo dõi đồng thời rất nhiều tín hiệu: quá tải công việc (OT cao, làm việc ngoài giờ), giảm sút năng suất, nghỉ phép đột xuất gia tăng, thời gian dài chưa được thăng chức, hoặc điểm gắn kết nội bộ sụt giảm.

Hệ thống được xây dựng nhằm giải quyết 4 câu hỏi cốt lõi:

1. **Xác suất nghỉ việc**: Tại một mốc snapshot date, nhân viên có xác suất thôi việc trong 90 ngày tới là bao nhiêu?
2. **Phân nhóm rủi ro (Risk Band)**: Nhân viên thuộc nhóm nào trong 4 mức: `LOW`, `MEDIUM`, `HIGH`, hay `CRITICAL`?
3. **Top Drivers**: Những chỉ số/tín hiệu nào đang kéo risk score tăng lên?
4. **Hành động & Đề xuất**: HR nên bắt đầu trao đổi với nhân viên từ chủ đề nào (workload, lộ trình sự nghiệp, đãi ngộ, hỗ trợ từ quản lý)?

> **Nguyên tắc đạo đức**: Mô hình là công cụ hỗ trợ quyết định (decision support), không dùng để kết luận nhân viên chắc chắn sẽ nghỉ, không dùng để phán xét hay gây áp lực.

---

## Kiến trúc hệ thống

```text
[Amazon RDS PostgreSQL (OLTP)]
        │
        ▼ (Full Load + CDC qua AWS DMS)
[Amazon S3 Raw Data Lake]
        │
        ▼ (AWS CLI / Ingestion Module + Manifest)
[Local / Training Environment]
        ├── data/raw/*.csv
        │
        ├── Module Python core (app/)
        │     ├── data.py          (merge_asof chống leakage)
        │     ├── features.py      (feature engineering & selection)
        │     ├── train.py         (baseline, ensemble, MLflow tracking)
        │     ├── evaluate.py      (PR-AUC, ROC-AUC, confusion matrix by risk band)
        │     └── explain.py       (top feature drivers & LLM reasoning guardrails)
        │
        ├── Notebook Demo (notebooks/hr_attrition_end_to_end.ipynb)
        │     └── Gọi trực tiếp từ app.* để thực thi và trực quan hoá
        │
        ├── CLI Scripts (scripts/*.py)
        │     └── Tự động hoá prepare data, training, evaluation, predict
        │
        ├── Model Artifacts (models/)
        │     └── best_model.joblib + metrics.json + schema.json
        │
        └── FastAPI Service (app/main.py)
              ├── GET  /health, /model-info, /model-metrics
              ├── POST /predict, /batch-predict
              ├── POST /explain (LLM reasoning with guardrails)
              └── POST /analysis-runs (PostgreSQL persistence)
```

---

## Cấu trúc dự án chuẩn

Cấu trúc được thiết kế theo tiêu chí: **Toàn bộ logic nghiệp vụ nằm trong module Python (`app/`), notebook chỉ import và hiển thị kết quả, script CLI phục vụ tự động hoá, FastAPI phục vụ serving**.

```text
hr-attrition-risk-prediction-v1/
├── data/
│   ├── raw/                              # Dữ liệu gốc sync từ S3 / RDS
│   │   ├── attrition_labels.csv          # Target & snapshot date
│   │   ├── employees.csv                 # Hồ sơ nhân viên (Master table)
│   │   ├── engagement_surveys.csv        # Khảo sát gắn kết định kỳ
│   │   ├── leave_logs.csv                # Lịch sử nghỉ phép
│   │   ├── performance_reviews.csv       # Đánh giá hiệu suất theo quý
│   │   ├── work_activity.csv             # Hoạt động làm việc hàng tuần
│   │   └── data_dictionary.csv           # Từ điển dữ liệu
│   ├── processed/                        # Dữ liệu trung gian sau tiền xử lý
│   │   ├── preprocessed.parquet
│   │   ├── feature_engineered.parquet
│   │   ├── train.parquet
│   │   └── test.parquet
│   └── samples/
│       ├── sample_single_employee.json   # Mẫu 1 nhân viên để test API
│       └── sample_batch_employees.json    # Mẫu danh sách nhân viên theo batch
│
├── notebooks/                            # Notebook phục vụ phân tích & báo cáo
│   └── 01_hr_attrition_end_to_end.ipynb  # End-to-end demo: Ingest -> EDA -> Feat -> Train -> Eval -> Explain -> Inference
│
├── models/                               # Artifacts sau khi train
│   ├── best_model.joblib                 # Pipeline hoàn chỉnh (Preprocess + Classifier)
│   ├── metrics.json                      # ROC-AUC, PR-AUC, F1, Confusion Matrix
│   ├── schema.json                       # Đặc tả feature schema phục vụ validation
│   └── production/
│       └── model_manifest.json           # Manifest phát hành model chính thức
│
├── reports/
│   ├── eda_charts/                       # Biểu đồ phân tích tương quan & EDA
│   └── evaluations/                      # Confusion matrix theo risk band, PR curve
│
├── scripts/                              # Các script độc lập chạy pipeline
│   ├── prepare_data.py                   # Ingest + tiền xử lý + feature engineering
│   ├── train_model.py                    # Huấn luyện đa mô hình + MLflow tracking
│   ├── evaluate_model.py                 # Đánh giá model trên test set & sinh báo cáo
│   ├── eda.py                            # Tự động xuất biểu đồ EDA
│   └── predict_sample.py                 # CLI dự đoán nhanh 1 file JSON đầu vào
│
├── app/                                  # Package mã nguồn chính (Clean Architecture)
│   ├── __init__.py
│   ├── config.py                         # Cấu hình biến môi trường (Pydantic BaseSettings)
│   ├── schemas.py                        # Pydantic schemas cho Request / Response API
│   ├── data.py                           # Đọc data, merge_asof chống leakage, validate schema
│   ├── features.py                       # Time-window aggregations & Feature Engineering
│   ├── train.py                          # Logic huấn luyện & so sánh 5 models
│   ├── evaluate.py                       # Metrics imbalance: PR-AUC, F1, Confusion Matrix by Risk Band
│   ├── explain.py                        # Feature Importance / SHAP drivers & LLM reasoning
│   ├── inference.py                      # Model loading, predict xác suất & gán Risk Band
│   ├── cli.py                            # CLI adapter cho Orchestration (Airflow / Cron)
│   └── main.py                           # FastAPI application & API routers
│
├── docs/                                 # Tài liệu kiến trúc & bằng chứng Lab Cloud
│   ├── DATASET_SCHEMA.md                 # Đặc tả chi tiết từng cột & kiểu dữ liệu
│   └── aws_dms_lab.md                    # Báo cáo & bằng chứng RDS -> DMS -> S3
│
├── tests/                                # Kiểm thử tự động
│   ├── test_schemas.py                   # Test validation schema Pydantic
│   ├── test_leakage.py                   # Test kiểm tra rò rỉ thời gian (leakage test)
│   └── test_inference.py                 # Test API endpoint & logic dự đoán
│
├── .env.example                          # Mẫu cấu hình môi trường
├── .gitignore
├── pyproject.toml
├── requirements.txt                      # Dependencies cho pipeline & training
├── requirements-api.txt                  # Dependencies nhẹ chỉ dành cho API serving
├── Dockerfile                            # Image phục vụ training & pipeline
├── Dockerfile.api                        # Image siêu nhẹ phục vụ FastAPI service
├── docker-compose.yml                    # Điều phối FastAPI + MLflow + PostgreSQL
└── README.md
```

---

## Cài đặt môi trường

### 1. Khởi tạo môi trường ảo (venv)

```bash
# Clone dự án và truy cập thư mục
cd hr-attrition-risk-prediction-v1

# Tạo virtual environment với Python 3.11+
python -m venv venv

# Kích hoạt môi trường
# Trên Linux/macOS:
source venv/bin/activate
# Trên Windows (PowerShell):
venv\Scripts\Activate.ps1
# Trên Windows (Command Prompt):
venv\Scripts\activate.bat

# Cài đặt thư viện đầy đủ
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Cấu hình biến môi trường

```bash
cp .env.example .env
```

Chỉnh sửa file `.env` theo nhu cầu (chạy local hoặc S3/MLflow):

```env
DATA_SOURCE=local                         # "local" hoặc "s3"
RAW_DATA_DIR=./data/raw
PROCESSED_DATA_DIR=./data/processed
MODELS_DIR=./models
REPORTS_DIR=./reports

# AWS S3 (Khi DATA_SOURCE=s3)
S3_BUCKET=your-hr-attrition-bucket
S3_PREFIX=raw/
AWS_DEFAULT_REGION=ap-southeast-1

# MLflow Tracking
MLFLOW_TRACKING_URI=http://localhost:5001
MLFLOW_EXPERIMENT_NAME=hr-attrition-risk-prediction-v1

# LLM Reasoning (Tùy chọn)
OPENAI_API_KEY=your-api-key-here
LLM_MODEL=gpt-4o-mini
```

---

## Hướng dẫn chạy Notebook (End-to-End Demo)

Theo đúng tiêu chuẩn đồ án: **Notebook không viết lại logic thuần, mà gọi trực tiếp các hàm chuẩn từ package `app/` để minh hoạ từng giai đoạn, hiển thị bảng số liệu và biểu đồ trực quan**.

### Khởi động Jupyter Notebook

```bash
jupyter notebook notebooks/01_hr_attrition_end_to_end.ipynb
```

### Luồng thực thi bên trong Notebook

```python
# 0. Import các module chuẩn từ app
from app.config import settings
from app.data import (sync_data_from_s3, validate_source_data, preprocess_data, feature_engineering, feature_selection, exploratory_data_analysis)
from app.train import split_data, handle_imbalance_data, train_and_track_models
from app.evaluate import evaluate_classifier, evaluate_risk_bands
from app.features import FINAL_FEATURES, TARGET_COLUMN
from app.inference import get_inference_services
from app.explain import generate_llm_explanation
from app.schemas import AttritionInput, ExplainInput

# 1. Ingest & Kiểm tra chất lượng dữ liệu
ingestion_report = sync_data_from_s3()
validation_report = validate_source_data()
display(pd.DataFrame([ingestion_report, validation_report]))

merged_df = pd.read_parquet(settings.MERGED_DATA_PATH)
display(merged_df.head())
print(f'Merged dataset shape: {merged_df.shape}')

# 2. EDA, Preprocessing & Feature Engineering
# Chỉ aggregate dữ liệu <= snapshot_date
eda_report = exploratory_data_analysis()
preprocessing_report = preprocess_data()
engineering_report = feature_engineering()
selection_report = feature_selection()
display(pd.DataFrame([eda_report, preprocessing_report, engineering_report, selection_report]))

feature_df = pd.read_parquet(settings.FEATURE_SELECTED_PATH)
display(feature_df.head())
print(f'Selected feature dataset shape: {feature_df.shape}')

# 3. Train & Đánh giá đa mô hình (MLflow tracking)
split_report = split_data()
balance_report = handle_imbalance_data()
display(pd.DataFrame([split_report, balance_report]))

train_df = pd.read_parquet(settings.TRAIN_BALANCED_PATH)
test_df = pd.read_parquet(settings.TEST_BALANCED_PATH)
training_summary = train_and_track_models(train_df, test_df)
metrics_df = pd.DataFrame(training_summary['full_metrics']).sort_values('roc_auc', ascending=False)
display(metrics_df)
print(f"Best model: {training_summary['best_model']['Model']}")

# 4. Đánh giá best model
pipeline = joblib.load(settings.BEST_MODEL_PATH)
X_test, y_test = test_df[FINAL_FEATURES], test_df[TARGET_COLUMN]
y_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = pipeline.predict(X_test)
metrics = evaluate_classifier(y_test, y_pred, y_proba)
risk_band_report = evaluate_risk_bands(y_test, y_proba)
display(pd.DataFrame([metrics]))
display(pd.DataFrame(risk_band_report['bands_summary']).T)

# 5. Explainability & LLM Reasoning có Guardrails
sample_row = test_df.dropna(subset=FINAL_FEATURES).iloc[0]
sample_payload = sample_row[FINAL_FEATURES].to_dict()
attrition_input = AttritionInput(**sample_payload)
prediction = get_inference_services().predict_one(attrition_input)
display(pd.DataFrame([prediction.model_dump(exclude={'top_drivers'})]))
display(pd.DataFrame([driver.model_dump() for driver in prediction.top_drivers]))

explain_input = ExplainInput(
    churn_probability=prediction.churn_probability,
    risk_band=prediction.risk_band,
    top_drivers=[driver.model_dump() for driver in prediction.top_drivers],
    department=sample_payload['department'],
    role=sample_payload['role'],
)
explanation = generate_llm_explanation(explain_input)
display(explanation)
```

---

## Chạy pipeline qua Script CLI

Nếu muốn tự động hoá pipeline mà không mở Notebook:

```bash
# Bước 1: Chuẩn bị dữ liệu (Ingest, validate, merge chống leakage, trích xuất đặc trưng)
python scripts/prepare_data.py

# Bước 2: Phân tích khám phá và tự động xuất biểu đồ vào reports/eda_charts/
python scripts/eda.py

# Bước 3: Huấn luyện 5 mô hình, log metrics và artifacts lên MLflow
python scripts/train_model.py

# Bước 4: Đánh giá mô hình tốt nhất với PR-AUC và Confusion Matrix theo Risk Band
python scripts/evaluate_model.py

# Bước 5: Thử nghiệm dự đoán nhanh cho 1 nhân viên từ file JSON mẫu
python scripts/predict_sample.py --input data/samples/sample_single_employee.json
```

---

## Triển khai FastAPI Inference Service

### 1. Chạy API ở môi trường Local

```bash
# Khởi động Uvicorn server với auto-reload
python -m uvicorn app.main:app --host 127.0.0.1 --port 8080 --reload
```

Truy cập Swagger UI tài liệu API:

```text
http://127.0.0.1:8080/docs
```

### 2. Chạy toàn bộ hệ thống bằng Docker Compose

Bao gồm: FastAPI app + MLflow Tracking Server + PostgreSQL Backend Database:

```bash
# Build và chạy ngầm
docker compose up --build -d

# Kiểm tra logs
docker compose logs -f api
```

Các cổng dịch vụ:

- **FastAPI**: `http://localhost:8080/docs`
- **MLflow UI**: `http://localhost:5001`
- **PostgreSQL**: `localhost:5433`

---

## Cơ chế chống rò rỉ dữ liệu (Anti-Leakage)

> **Nguyên tắc vàng**: Khi dự đoán tại mốc `snapshot_date`, tuyệt đối không sử dụng bất kỳ thông tin nào phát sinh sau thời điểm này.

Hệ thống tuân thủ 3 lớp bảo vệ:

1. **Loại bỏ nhãn sau sự kiện**: Cột `termination_date` và `attrition_type` trong `attrition_labels.csv` chỉ xuất hiện sau khi nhân viên đã nghỉ. Chúng được loại bỏ ngay trong bước đọc dữ liệu:
   ```python
   labels = labels.drop(columns=["termination_date", "attrition_type"], errors="ignore")
   ```
2. **Temporal Backward Join (`pd.merge_asof`)**: Khi ghép nối các bảng chuỗi thời gian (`work_activity`, `leave_logs`, `performance_reviews`, `engagement_surveys`), hệ thống chỉ lấy các bản ghi có ngày phát sinh $\le$ `snapshot_date`:
   ```python
   merged = pd.merge_asof(
       labels.sort_values("snapshot_date"),
       activity.sort_values("week_start"),
       left_on="snapshot_date",
       right_on="week_start",
       by="employee_id",
       direction="backward"  # Chỉ lấy thông tin trong quá khứ hoặc hiện tại
   )
   ```
3. **Time-based Train/Test Split**: Thay vì chia ngẫu nhiên (dễ bị rò rỉ quan sát giữa các quý), hệ thống chia train/test theo mốc thời gian: dữ liệu trước mốc thời gian chốt làm Train, các quý sau làm Test.

---

## Đặc tả chỉ số Imbalance Data & Risk Bands

Trong bài toán nghỉ việc, số lượng nhân viên rời đi thường chiếm tỷ lệ rất nhỏ (~4.29% trong tập dữ liệu), dẫn đến mất cân bằng lớp nghiêm trọng:

### 1. Các chỉ số đo lường chính

| Chỉ số                            | Lý do áp dụng                                                                                                |
| :-------------------------------- | :----------------------------------------------------------------------------------------------------------- |
| **ROC-AUC**                       | Đánh giá tổng quát năng lực xếp hạng xác suất của mô hình độc lập với ngưỡng cut-off                         |
| **PR-AUC (Precision-Recall AUC)** | **Chỉ số quan trọng nhất cho dữ liệu cực kỳ mất cân bằng**, tập trung trực tiếp vào lớp thiểu số (nghỉ việc) |
| **Recall (Sensitivity)**          | Đo lường tỷ lệ phát hiện đúng nhân viên thực sự có ý định nghỉ (tránh bỏ sót)                                |
| **Precision**                     | Đo lường độ tin cậy của cảnh báo (tránh làm phiền nhân viên và lãng phí thời gian của HR)                    |
| **F1-Score**                      | Trung bình điều hòa giữa Precision và Recall                                                                 |

### 2. Phân loại 4 dải rủi ro (Risk Bands) & Hành động đề xuất

Thay vì chỉ trả về 0/1 theo ngưỡng cứng 0.5, hệ thống chia xác suất thành 4 phân khúc nghiệp vụ rõ ràng:

| Xác suất (`churn_probability`) | Risk Band  | Ý nghĩa nghiệp vụ                                  | Hành động khuyến nghị cho HR                                                                            |
| :----------------------------: | :--------: | :------------------------------------------------- | :------------------------------------------------------------------------------------------------------ |
|         **$\ge 0.75$**         | `CRITICAL` | Nguy cơ nghỉ việc cực kỳ khẩn cấp trong 30-60 ngày | Lên lịch gặp 1-1 khẩn cấp trong tuần; rà soát chế độ lương thưởng; chuẩn bị phương án giữ chân đặc biệt |
|       **$0.50 - 0.74$**        |   `HIGH`   | Rủi ro cao, có nhiều tín hiệu bất ổn rõ rệt        | Trao đổi với quản lý trực tiếp; tìm hiểu vướng mắc về khối lượng công việc và định hướng phát triển     |
|       **$0.25 - 0.49$**        |  `MEDIUM`  | Dấu hiệu cảnh báo sớm, cần theo dõi thêm           | Khảo sát mức độ hài lòng; kiểm tra cân bằng công việc - đời sống qua các kỳ đánh giá tiếp theo          |
|          **$< 0.25$**          |   `LOW`    | Ổn định, gắn bó tốt                                | Duy trì lộ trình đào tạo và môi trường làm việc bình thường                                             |

### 3. Confusion Matrix theo Risk Band

Bảng ma trận chéo giữa **Thực tế nghỉ việc (0/1)** và **4 dải Risk Band** giúp HR biết được trong số các ca `CRITICAL` và `HIGH` có bao nhiêu % nhân viên thực sự nghỉ việc.

---

## Cơ chế Explainability & LLM Reasoning có Guardrails

Sau khi mô hình đưa ra `churn_probability` và `risk_band`, hệ thống trích xuất **Top 3 Drivers** có tác động kéo rủi ro lên cao nhất (dựa trên Feature Importance / SHAP value). LLM nhận dữ liệu này để sinh văn bản giải thích cho HR.

### 1. Guardrails nghiêm ngặt cho LLM

- **Không suy diễn ngoài dữ liệu**: LLM chỉ được dựa vào các biến số được cung cấp (số giờ OT, điểm survey, ngày nghỉ đột xuất...). Không tự bịa lý do đời tư, tính cách hay thái độ.
- **Không phán xét hay khẳng định tuyệt đối**: Cấm viết _"Nhân viên này chắc chắn sẽ nghỉ việc"_ hoặc _"Cần sa thải / thay thế nhân viên này"_.
- **Tập trung vào giải pháp hỗ trợ**: Văn phong mang tính đồng hành, gợi ý các câu hỏi mở để quản lý và HR lắng nghe tâm tư nhân viên.

### 2. Ví dụ so sánh Output

#### ❌ Phản hồi vi phạm (Không có Guardrails):

> _"Nhân viên E0123 chắc chắn sắp nghỉ việc vì không còn động lực cống hiến. Công ty nên chuẩn bị tuyển người mới thay thế vị trí này ngay lập tức."_

#### ✅ Phản hồi chuẩn mực (Đã áp dụng Guardrails):

> _"Nhân viên đang ở mức rủi ro **HIGH (Xác suất: 71.5%)**. Các yếu tố chính làm tăng rủi ro gồm: số giờ làm thêm 4 tuần gần nhất tăng cao (trung bình 8.2 giờ/tuần), số ngày nghỉ đột xuất trong quý tăng (4 ngày), và điểm khảo sát cân bằng công việc giảm 25 điểm so với kỳ trước. Khuyến nghị HR nên trao đổi với tinh thần hỗ trợ: lắng nghe về áp lực tiến độ hiện tại, rà soát phân bổ công việc cùng quản lý trực tiếp, và đánh giá tính linh hoạt trong thời gian làm việc."_

---

## Đặc tả API Contract (FastAPI)

### 1. `GET /health`

Kiểm tra tính sẵn sàng của service và model.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "XGBoostClassifier",
  "version": "1.0.0",
  "timestamp": "2026-10-03T08:00:00Z"
}
```

---

### 2. `GET /model-info`

Xem thông tin kiến trúc mô hình, danh sách thuộc tính đầu vào và phiên bản đang phục vụ.

**Response:**

```json
{
  "model_name": "XGBoost",
  "model_type": "Pipeline(ColumnTransformer + XGBClassifier)",
  "features_count": 32,
  "numeric_features": [
    "tasks_completed",
    "late_tasks",
    "overtime_hours_4w_avg",
    "unplanned_leave_days_90d",
    "engagement_score",
    "tenure_months"
  ],
  "categorical_features": [
    "department",
    "role",
    "job_level",
    "location",
    "employment_type"
  ],
  "quality_gate": {
    "metric": "ROC-AUC",
    "threshold": 0.7,
    "current_value": 0.9534
  }
}
```

---

### 3. `GET /model-metrics`

Xem bảng kết quả đánh giá chi tiết trên tập kiểm thử độc lập.

**Response:**

```json
{
  "evaluation_date": "2026-10-02T17:00:00Z",
  "metrics": {
    "roc_auc": 0.9534,
    "pr_auc": 0.8241,
    "f1_score": 0.8049,
    "precision": 0.9206,
    "recall": 0.7151
  },
  "risk_band_distribution": {
    "CRITICAL": 142,
    "HIGH": 380,
    "MEDIUM": 810,
    "LOW": 4200
  }
}
```

---

### 4. `POST /predict`

Dự đoán rủi ro nghỉ việc, xếp dải Risk Band, trích xuất Top Drivers và gợi ý hành động cho 1 nhân viên.

**Request Body:**

```json
{
  "employee_id": "E004128",
  "features": {
    "tasks_completed": 42,
    "late_tasks": 4,
    "overtime_hours_4w_avg": 7.5,
    "meeting_hours": 6.0,
    "focus_hours": 12.0,
    "after_hours_messages": 10,
    "absent_days": 1,
    "performance_score": 3.8,
    "goal_completion_rate": 0.82,
    "manager_rating": 3.5,
    "peer_feedback_score": 3.9,
    "review_sentiment_score": 0.65,
    "leave_days": 3.0,
    "unplanned_leave_days_90d": 3,
    "manager_support_score": 3.0,
    "career_growth_score": 2.5,
    "compensation_satisfaction_score": 2.8,
    "engagement_score": 45.0,
    "work_life_balance_score": 40.0,
    "job_level": 2,
    "monthly_salary_usd": 3200,
    "salary_band_min_usd": 2800,
    "salary_band_max_usd": 4000,
    "promotion_count": 0,
    "tenure_months": 22.0,
    "days_since_last_promotion": 660,
    "leave_type": "Sick",
    "is_unplanned": true,
    "approved_by_manager": true,
    "department": "Engineering",
    "role": "Software Engineer",
    "location": "Ho Chi Minh City",
    "employment_type": "Full-time"
  }
}
```

**Response Body:**

```json
{
  "employee_id": "E004128",
  "churn_probability": 0.724,
  "churn_prediction": 1,
  "risk_band": "HIGH",
  "top_drivers": [
    {
      "feature": "overtime_hours_4w_avg",
      "value": 7.5,
      "impact": "Tăng rủi ro (làm thêm giờ liên tục)"
    },
    {
      "feature": "career_growth_score",
      "value": 2.5,
      "impact": "Tăng rủi ro (kỳ vọng phát triển thấp)"
    },
    {
      "feature": "days_since_last_promotion",
      "value": 660,
      "impact": "Tăng rủi ro (thời gian chưa thăng chức dài)"
    }
  ],
  "recommended_action": "Trao đổi với quản lý trực tiếp; tìm hiểu vướng mắc về khối lượng công việc và định hướng phát triển sự nghiệp.",
  "model_version": "xgb-2026-10"
}
```

---

### 5. `POST /explain`

Nhận kết quả tính toán từ mô hình và gọi LLM reasoning sinh đoạn văn giải thích có guardrail phục vụ HR.

**Request Body:**

```json
{
  "employee_id": "E004128",
  "churn_probability": 0.724,
  "risk_band": "HIGH",
  "department": "Engineering",
  "role": "Software Engineer",
  "top_drivers": [
    { "feature": "overtime_hours_4w_avg", "value": 7.5 },
    { "feature": "career_growth_score", "value": 2.5 },
    { "feature": "days_since_last_promotion", "value": 660 }
  ]
}
```

**Response Body:**

```json
{
  "employee_id": "E004128",
  "explanation_text": "Nhân viên đang ở mức rủi ro HIGH với xác suất nghỉ việc là 72.4%. Các tín hiệu chính từ dữ liệu cho thấy nhân viên đang có số giờ làm thêm trung bình 4 tuần qua ở mức cao (7.5h/tuần), điểm đánh giá cơ hội phát triển sự nghiệp thấp (2.5/5), và đã gần 2 năm chưa có thăng chức mới. HR nên ưu tiên trao đổi cùng quản lý bộ phận để lắng nghe về khối lượng công việc hiện tại và làm rõ lộ trình thăng tiến của nhân viên.",
  "suggested_topics": [
    "Workload & Cân bằng thời gian",
    "Kế hoạch nâng bậc và phát triển chuyên môn",
    "Sự hỗ trợ từ quản lý trực tiếp"
  ],
  "guardrail_status": "passed"
}
```

---

### 6. `POST /batch-predict`

Nhận danh sách nhiều nhân viên, xử lý đồng thời và trả về bảng xếp hạng ưu tiên theo mức rủi ro giảm dần.

---

### 7. `POST /analysis-runs` (Lưu trữ kết quả vào PostgreSQL)

Lưu toàn bộ phiên chấm điểm batch cùng các trường dữ liệu và kết quả vào cơ sở dữ liệu nếu biến `DATABASE_URL` được cấu hình. Nếu chưa có database, hệ thống trả về `saved_to_database: false` mà không gây gián đoạn API.

---

## Bằng chứng Lab Cloud (AWS RDS -> DMS -> S3)

Theo yêu cầu đồ án, dữ liệu nguồn được trích xuất từ database tác nghiệp qua Data Lake:

1. **Source**: RDS PostgreSQL database `hr_attrition` lưu trữ 6 bảng quan hệ.
2. **Migration Tool**: AWS Database Migration Service (DMS) chạy tác vụ **Full load + CDC** (Change Data Capture).
3. **Target**: S3 Bucket raw zone (`s3://your-hr-attrition-bucket/raw/`) lưu trữ dưới dạng parquet/csv.
4. **Kiểm thử CDC**: Thực hiện thêm, sửa, xóa bản ghi trên RDS và xác nhận file delta được ghi nhận tức thời trên S3.
5. Tài liệu hướng dẫn và ảnh chụp minh chứng chi tiết được lưu trữ tại: [`docs/aws_dms_lab.md`](docs/aws_dms_lab.md).

---

## Bộ biến môi trường hoàn chỉnh

| Tên biến                 |  Kiểu  |        Mặc định         | Ý nghĩa & Mô tả                                          |
| :----------------------- | :----: | :---------------------: | :------------------------------------------------------- |
| `DATA_SOURCE`            | string |         `local`         | Chọn nguồn dữ liệu: `local` (thư mục) hoặc `s3` (AWS S3) |
| `RAW_DATA_DIR`           | string |      `./data/raw`       | Đường dẫn lưu các file CSV đầu vào tại máy local         |
| `PROCESSED_DATA_DIR`     | string |   `./data/processed`    | Thư mục lưu dữ liệu sạch dạng Parquet                    |
| `MODELS_DIR`             | string |       `./models`        | Thư mục lưu model binary `.joblib` và schema metadata    |
| `S3_BUCKET`              | string |          `""`           | Tên S3 Bucket chứa dữ liệu sau khi DMS đồng bộ           |
| `S3_PREFIX`              | string |         `raw/`          | Tiền tố thư mục trong S3 Bucket                          |
| `AWS_DEFAULT_REGION`     | string |    `ap-southeast-1`     | Vùng triển khai AWS                                      |
| `AWS_ACCESS_KEY_ID`      | string |          `""`           | Access Key kết nối S3                                    |
| `AWS_SECRET_ACCESS_KEY`  | string |          `""`           | Secret Key kết nối S3                                    |
| `MIN_ROC_AUC`            | float  |         `0.70`          | Ngưỡng Quality Gate (ROC-AUC tối thiểu để publish)       |
| `MLFLOW_TRACKING_URI`    | string | `http://localhost:5001` | Địa chỉ server MLflow Tracking                           |
| `MLFLOW_EXPERIMENT_NAME` | string |   `hr-attrition-risk`   | Tên thí nghiệm trên MLflow                               |
| `DATABASE_URL`           | string |          `""`           | Chuỗi kết nối PostgreSQL lưu kết quả `/analysis-runs`    |
| `OPENAI_API_KEY`         | string |          `""`           | Khóa API dịch vụ OpenAI cho module LLM reasoning         |

---

## Lộ trình phát triển & Mở rộng (Next Steps)

- [x] **Data Ingestion & Anti-Leakage**: Xây dựng hàm kết hợp chuỗi thời gian chặt chẽ với `pd.merge_asof`.
- [x] **Experiment Tracking**: Tích hợp MLflow nested runs cho 5 thuật toán Machine Learning.
- [x] **FastAPI Service**: Thiết kế API chuẩn RESTful với Pydantic schema validation và 4 dải Risk Band.
- [x] **Notebook Demo**: Tạo notebook trực quan hoá toàn bộ pipeline mà không chứa logic thừa.
- [ ] **SHAP Integration**: Thay thế Tree Feature Importance bằng SHAP TreeExplainer cho từng cá nhân.
- [ ] **LangChain / LlamaIndex Guardrails**: Tích hợp NeMo Guardrails để kiểm duyệt đầu ra của LLM tự động.
- [ ] **Frontend Dashboard**: Xây dựng giao diện Next.js / Streamlit cho phòng Nhân sự tra cứu trực quan.
- [ ] **CI/CD Automation**: GitHub Actions tự động kiểm thử unit test, linting và build Docker image khi push code.
