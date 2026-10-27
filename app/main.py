from datetime import datetime, timezone
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.inference import get_inference_services
from app.explain import get_explanation_service
from app.schemas import (
    AttritionInput,
    AttritionPrediction,
    BatchAttritionInput,
    BatchAttritionPrediction,
    HealthResponse,
    ModelInfoResponse,
    ExplainInput,
    ExplainResponse,
)


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Khởi động: Thử nạp model trước nếu có sẵn
#     try:
#         get_model()
#         print("Model loaded successfully at startup.")
#     except Exception as e:
#         print(f"Warning: Model not pre-loaded ({e}). Will load on first request.")
#     yield


app = FastAPI(
    title="HR Attrition Risk Prediction & LLM Reasoning API",
    description="API dự đoán rủi ro nghỉ việc của nhân viên trong 90 ngày kết hợp LLM Reasoning có Guardrails.",
    version="1.0.0",
    # lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Kiểm tra trạng thái sẵn sàng của dịch vụ và model."""
    service = get_inference_services()
    return HealthResponse(
        status="ok",
        model_loaded=True if service.model_path else False,
        model_version=service.model_version if service.model_path else None
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    """Lấy thông tin kiến trúc, features và ngưỡng chất lượng của model."""
    if not settings.METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Feature schema not found. Run training first.")

    metrics_data = json.loads(settings.METRICS_PATH.read_text())
    service = get_inference_services()
    best_metrics = metrics_data.get("best_model", {})
    return ModelInfoResponse(
        model_name=service.pipeline.__class__.__name__,
        model_path=str(settings.BEST_MODEL_PATH),
        published_at=datetime.now(timezone.utc).isoformat(),
        numeric_features=service.numeric_features,
        categorical_features=service.categorical_features,
        accuracy=best_metrics.get("accuracy", 0.0),
        precision=best_metrics.get("precision", 0.0),
        recall=best_metrics.get("recall", 0.0),
        f1_score=best_metrics.get("f1", 0.0),
        roc_auc=best_metrics.get("roc_auc", 0.0)
    )

@app.post("/predict", response_model=AttritionPrediction, tags=["Inference"])
def predict(attrition: AttritionInput) -> AttritionPrediction:
    service = get_inference_services()
    return service.predict_one(attrition)


@app.post("/batch_predict", response_model=BatchAttritionPrediction, tags=["Inference"])
def batch_predict(payload: BatchAttritionInput) -> BatchAttritionPrediction:
    service = get_inference_services()
    return BatchAttritionPrediction(predictions=service.predict_many(attrition_lists=payload.attrition_list))


@app.post("/explain", response_model=ExplainResponse, tags=["LLM Reasoning"])
def explain_attrition(payload: ExplainInput) -> ExplainResponse:
    """Tạo đoạn văn bản giải thích bằng LLM có Guardrails dựa trên kết quả tính toán."""
    service = get_explanation_service()
    return service.generate_llm_explanation(payload)

# @app.get("/ingest_data", response_model=IngestDataResponse, tags=["Data Pipeline"])
# def ingest_data() -> IngestDataResponse:
#     """Kéo dữ liệu từ S3, xử lý và lưu trữ vào thư mục data/raw. Trả về báo cáo chi tiết."""
#     service = get_data_service()
#     return service.ingest_data()

# @app.get("/validate_data", response_model=dict, tags=["Data Pipeline"])
# def validate_data() -> dict:
#     """Kiểm tra dữ liệu đã ingest, trả về báo cáo chi tiết."""
#     service = get_data_service()
#     return service.validate_data()

# @app.post("/analysis-runs", response_model=AnalysisRunsResponse, tags=["Persistence"])
# def create_analysis_run(request: AnalysisRunsRequest):
#     """Chấm điểm batch và lưu trữ kết quả phân tích vào PostgreSQL (nếu được cấu hình)."""
#     run_id = str(uuid.uuid4())
#     predictions = batch_predict_employees(request.employees)

#     saved = False
#     status_msg = "DATABASE_URL is not configured; analysis was processed in-memory."

#     if settings.DATABASE_URL:
#         try:
#             # Ghi vào PostgreSQL nếu có database connection
#             # Tự động tạo bảng nếu cần
#             status_msg = "Successfully persisted analysis run and employee scores to database."
#             saved = True
#         except Exception as e:
#             status_msg = f"Failed to persist to database: {str(e)}"

#     return AnalysisRunsResponse(
#         analysis_run_id=run_id,
#         saved_to_database=saved,
#         persistence_status=status_msg,
#         total_processed=len(predictions),
#         results=predictions
#     )
