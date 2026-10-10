from typing import Any, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class AttritionInput(BaseModel):
    # Numeric features (25)
    tasks_completed: float = Field(..., description="Số lượng task hoàn thành gần nhất", examples=[61])
    late_tasks: float = Field(..., description="Số task trễ hạn", examples=[3])
    overtime_hours: float = Field(..., description="Số giờ OT", examples=[5])
    meeting_hours: float = Field(..., description="Số giờ họp", examples=[2])
    focus_hours: float = Field(..., description="Số giờ làm việc tập trung", examples=[4])
    after_hours_messages: float = Field(..., description="Số tin nhắn gửi ngoài giờ", examples=[10])
    absent_days: float = Field(..., description="Số ngày vắng mặt", examples=[1])
    performance_score: float = Field(..., description="Điểm hiệu năng (1.0 - 5.0)", examples=[4.5])
    goal_completion_rate: float = Field(..., description="Tỷ lệ hoàn thành mục tiêu (0.0 - 1.0)", examples=[0.8])
    manager_rating: float = Field(..., description="Đánh giá từ quản lý (1.0 - 5.0)", examples=[4])
    peer_feedback_score: float = Field(..., description="Điểm phản hồi từ đồng nghiệp (1.0 - 5.0)", examples=[4])
    review_sentiment_score: float = Field(..., description="Điểm sentiment đánh giá (0.0 - 1.0)", examples=[0.9])
    leave_days: float = Field(..., description="Số ngày nghỉ phép", examples=[2])
    manager_support_score: float = Field(..., description="Độ hài lòng hỗ trợ từ sếp (10 - 100)", examples=[80])
    career_growth_score: float = Field(..., description="Độ hài lòng cơ hội thăng tiến (10 - 100)", examples=[70])
    compensation_satisfaction_score: float = Field(..., description="Độ hài lòng đãi ngộ (10 - 100)", examples=[90])
    job_level: int = Field(..., description="Cấp bậc chuyên môn (1 - 5)", examples=[3])
    salary_band_min_usd: float = Field(..., description="Lương min của band", examples=[1542])
    salary_band_max_usd: float = Field(..., description="Lương max của band", examples=[3782])
    monthly_salary_usd: float = Field(..., description="Mức lương tháng thực tế", examples=[1570])
    promotion_count: int = Field(..., description="Số lần thăng chức", examples=[1])
    engagement_score: float = Field(..., description="Điểm gắn kết tổng thể (10 - 100)", examples=[75])
    work_life_balance_score: float = Field(..., description="Điểm cân bằng công việc - đời sống (10 - 100)", examples=[80])
    tenure_months: float = Field(..., description="Thâm niên tính bằng tháng", examples=[24])
    days_since_last_promotion: float = Field(..., description="Số ngày kể từ lần thăng chức gần nhất", examples=[30])

    # Categorical features
    leave_type: str = Field(..., description="Loại nghỉ phép: Annual, Sick, Unpaid, Personal", examples=["Annual"])
    is_unplanned: bool = Field(..., description="Nghỉ đột xuất không lên lịch trước", examples=[True])
    approved_by_manager: bool = Field(..., description="Quản lý đã duyệt nghỉ", examples=[True])
    department: str = Field(..., description="Phòng ban: Engineering, Sales, Customer Success...", examples=["Customer Success"])
    role: str = Field(..., description="Chức danh công việc", examples=["Software Engineer"])
    location: str = Field(..., description="Địa điểm làm việc: Ho Chi Minh City, Ha Noi...", examples=["Ho Chi Minh City"])
    employment_type: str = Field(..., description="Hình thức làm việc: Full-time, Part-time, Contract")

class FeatureDriver(BaseModel):
    feature: str
    value: Any
    impact: str

class AttritionPrediction(BaseModel):
    churn_probability: float = Field(..., description="Xác suất nghỉ việc trong 90 ngày tới")
    risk_band: str = Field(..., description="LOW, MEDIUM, HIGH, hoặc CRITICAL")
    top_drivers: List[FeatureDriver] = Field(default_factory=list, description="Top đặc trưng làm tăng rủi ro")
    recommended_action: str = Field(..., description="Hành động khuyến nghị cho HR")
    model_name: str
    model_version: str

class BatchAttritionInput(BaseModel):
    attrition_list: List[AttritionInput]

class BatchAttritionPrediction(BaseModel):
    predictions: List[AttritionPrediction]

class ModelInfoResponse(BaseModel):
    model_name: str
    model_path: str
    published_at: Optional[str] = None
    numeric_features: List[str]
    categorical_features: List[str]
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None

class ExplainInput(BaseModel):
    churn_probability: float = Field(..., description="Xác suất nghỉ việc trong 90 ngày tới", examples=[0.8425])
    risk_band: str = Field(..., description="Nhóm rủi ro: LOW, MEDIUM, HIGH, hoặc CRITICAL", examples=["HIGH"])
    top_drivers: List[dict] = Field(
        ...,
        description="Danh sách các đặc trưng (features) làm tăng rủi ro nghỉ việc",
        examples=[[{"feature": "overtime_hours", "value": 15, "impact": "Làm việc quá giờ liên tục"}]]
    )
    department: Optional[str] = Field("N/A", description="Phòng ban của nhân viên (tuỳ chọn)", examples=["Engineering"])
    role: Optional[str] = Field("N/A", description="Chức danh của nhân viên (tuỳ chọn)", examples=["Software Engineer"])

    @field_validator("top_drivers", mode="before")
    @classmethod
    def ensure_list(cls, v):
        if isinstance(v, dict):
            return [v]
        return v

class ExplainResponse(BaseModel):
    explanation_text: str = Field(..., description="Đoạn văn bản giải thích cho HR")
    suggested_topics: List[str] = Field(..., description="Các chủ đề gợi ý cho HR")
    guardrail_status: Literal["passed", "failed"] = Field(..., description="Trạng thái tuân thủ nguyên tắc (guardrail)")

class LLMConfig(BaseModel):
    provider: Literal["openai", "anthropic", "groq", "local"] = Field(..., description="Nhà cung cấp LLM (ví dụ: openai, anthropic, ...)")
    model_name: str = Field(..., description="Tên model LLM (ví dụ: gpt-4o-mini)")
    api_key: Optional[str] = Field(None, description="Khóa API cho nhà cung cấp LLM")
    api_url: Optional[str] = Field(None, description="URL endpoint cho nhà cung cấp LLM") 

class IngestDataResponse(BaseModel):
    status: str = Field(..., description="Trạng thái xử lý dữ liệu")
    bucket: str = Field(..., description="Tên bucket lưu trữ file CSV")
    prefix: str = Field(..., description="Tiền tố (prefix) lưu trữ file CSV")
    files: List[str] = Field(..., description="Danh sách các file CSV đã lưu trữ")  
    data_dir: str = Field(..., description="Thư mục lưu trữ dữ liệu đã xử lý")

class ValidateDataResponse(BaseModel):
    status: str = Field(..., description="Trạng thái kiểm tra dữ liệu")
    validated_at: str = Field(..., description="Thời điểm kiểm tra dữ liệu")
    rows: int = Field(..., description="Số lượng bản ghi trong dữ liệu")
    columns: int = Field(..., description="Số lượng cột trong dữ liệu")
    source_files: List[str] = Field(..., description="Danh sách các file nguồn đã kiểm tra")

# class PredictRequest(BaseModel):
#     employee_id: Optional[str] = Field("EMP_UNKNOWN", description="Mã nhân viên (tuỳ chọn)")
#     features: EmployeeFeatures


# class PredictResponse(BaseModel):
#     employee_id: str
#     churn_probability: float = Field(..., description="Xác suất nghỉ việc trong 90 ngày tới")
#     churn_prediction: int = Field(..., description="1 nếu có rủi ro cao, 0 nếu an toàn")
#     risk_band: str = Field(..., description="LOW, MEDIUM, HIGH, hoặc CRITICAL")
#     top_drivers: List[FeatureDriver] = Field(default_factory=list, description="Top đặc trưng làm tăng rủi ro")
#     recommended_action: str = Field(..., description="Hành động khuyến nghị cho HR")
#     model_name: str
#     model_version: str


# class EmployeeBatchItem(BaseModel):
#     employee_id: str
#     features: EmployeeFeatures


# class BatchPredictRequest(BaseModel):
#     employees: List[EmployeeBatchItem]


# class BatchPredictResponse(BaseModel):
#     total: int
#     results: List[PredictResponse]
#     model_name: str
#     model_version: str


# class ExplainRequest(BaseModel):
#     employee_id: str
#     churn_probability: float
#     risk_band: str
#     department: Optional[str] = "N/A"
#     role: Optional[str] = "N/A"
#     top_drivers: List[dict]


# class ExplainResponse(BaseModel):
#     employee_id: str
#     explanation_text: str
#     suggested_topics: List[str]
#     guardrail_status: str


# class HealthResponse(BaseModel):
#     status: str
#     model_loaded: bool
#     model_name: str
#     timestamp: str


# class ModelInfoResponse(BaseModel):
#     model_name: str
#     model_path: str
#     published_at: Optional[str] = None
#     numeric_features: List[str]
#     categorical_features: List[str]
#     quality_gate_roc_auc: float


# class ModelMetricsResponse(BaseModel):
#     best_model: dict
#     full_metrics: List[dict]


# class EmployeeAnalysisRecord(BaseModel):
#     employee_id: str
#     employee_name: Optional[str] = None
#     department: Optional[str] = None
#     role: Optional[str] = None
#     manager: Optional[str] = None
#     scenario_note: Optional[str] = None
#     source_row: Optional[int] = None
#     features: EmployeeFeatures


# class AnalysisRunsRequest(BaseModel):
#     source_filename: str = "batch_employees.json"
#     source_note: Optional[str] = "Ad-hoc analysis batch"
#     employees: List[EmployeeAnalysisRecord]


# class AnalysisRunsResponse(BaseModel):
#     analysis_run_id: str
#     saved_to_database: bool
#     persistence_status: str
#     total_processed: int
#     results: List[PredictResponse]
