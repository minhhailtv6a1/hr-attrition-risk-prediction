import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from functools import lru_cache
from app.config import settings
from app.schemas import FeatureDriver, ExplainInput, ExplainResponse, LLMConfig

def extract_feature_drivers(
    model: Any,
    features_dict: Dict[str, Any],
    top_k: int = 3
) -> List[FeatureDriver]:
    """
    Trích xuất top feature drivers làm tăng nguy cơ nghỉ việc dựa trên các chỉ số
    nghiệp vụ chính (overtime, thâm niên chưa thăng chức, engagement, review...).
    """
    drivers = []

    # 1. Overtime
    ot = float(features_dict.get("overtime_hours", 0))
    if ot >= 5.0:
        drivers.append(FeatureDriver(
            feature="overtime_hours",
            value=ot,
            impact=f"Làm thêm giờ ở mức cao ({ot}h/tuần) có nguy cơ gây quá tải"
        ))

    # 2. Days since last promotion
    dsp = float(features_dict.get("days_since_last_promotion", 0))
    if dsp >= 500:
        drivers.append(FeatureDriver(
            feature="days_since_last_promotion",
            value=dsp,
            impact=f"Thời gian chưa được thăng chức dài ({int(dsp)} ngày)"
        ))

    # 3. Engagement score
    eng = float(features_dict.get("engagement_score", 50))
    if eng <= 45.0:
        drivers.append(FeatureDriver(
            feature="engagement_score",
            value=eng,
            impact=f"Điểm gắn kết ở mức thấp ({eng}/100)"
        ))

    # 4. Work life balance
    wlb = float(features_dict.get("work_life_balance_score", 50))
    if wlb <= 45.0:
        drivers.append(FeatureDriver(
            feature="work_life_balance_score",
            value=wlb,
            impact=f"Điểm cân bằng công việc - đời sống thấp ({wlb}/100)"
        ))

    # 5. Unplanned leaves
    unplanned = bool(features_dict.get("is_unplanned", False))
    leave_days = float(features_dict.get("leave_days", 0))
    if unplanned and leave_days >= 2:
        drivers.append(FeatureDriver(
            feature="unplanned_leave",
            value=leave_days,
            impact="Có đợt nghỉ đột xuất không lên lịch trước"
        ))

    # Fallback nếu không có tín hiệu vượt ngưỡng
    if not drivers:
        drivers.append(FeatureDriver(
            feature="general_indicators",
            value="Ổn định",
            impact="Các chỉ số tổng thể ở mức trung bình, cần duy trì theo dõi định kỳ"
        ))

    return drivers[:top_k]



    # ==============================================================================
    # 2. Prompt Engineering for Structured Output
    # ==============================================================================
HR_REASONING_SYSTEM_PROMPT = """
Bạn là chuyên gia tư vấn People Analytics nội bộ cao cấp, hỗ trợ các HR Business Partner (HRBP) và Quản lý trực tiếp (Line Manager) giải thích rủi ro nghỉ việc của nhân viên một
cách khách quan, thấu cảm và có trách nhiệm.

Nhiệm vụ của bạn là nhận các tín hiệu dự báo từ mô hình Machine Learning và tạo ra phản hồi có cấu trúc
(Structured Output) theo đúng 3 trường dữ liệu sau:

1. `explanation_text` (Chuỗi văn bản giải thích):
    - Độ dài: 3 đến 4 câu ngắn gọn bằng tiếng Việt chuyên nghiệp, tự nhiên.
    - Nội dung: Tóm tắt mức độ rủi ro và giải thích nguyên nhân dựa TRỰC TIẾP và DUY NHẤT vào các đặc trưng rủi ro (top drivers) được cung cấp.
    - Giọng điệu: Khách quan, thấu cảm, mang tính xác suất (ví dụ: "chỉ số cho thấy dấu hiệu rủi ro", "ghi nhận mức độ quá tải gần đây").
    - Tuyên bố miễn trừ trách nhiệm (Disclaimer): Luôn kèm một câu nhắc rõ đây là chỉ số cảnh báo sớm để hỗ trợ người quản lý lắng nghe và can thiệp kịp thời, KHÔNG PHẢI là kết luận chắc chắn về quyết định hay ý định cá nhân của nhân viên.

    2. `suggested_topics` (Mảng gồm 2 đến 3 chủ đề thảo luận 1-on-1):
    - Đưa ra chính xác 2 đến 3 chủ đề mở mang tính hỗ trợ, xây dựng và đồng hành (ví dụ: "Khối lượng công việc và cân bằng thời gian", "Định hướng phát triển và ghi nhận nỗ lực", "Kế hoạch nâng bậc và lộ trình sự nghiệp").
    - Tuyệt đối KHÔNG dùng câu hỏi chất vấn, đối đầu hoặc nghi ngờ nhân viên.

    3. `guardrail_status` (Chỉ nhận "passed" hoặc "failed"):
    - Tự kiểm toán (Self-audit) mức độ tuân thủ các nguyên tắc đạo đức nhân sự:
        + Đặt là "passed" nếu: Phản hồi hoàn toàn dựa trên dữ liệu có thật, giọng điệu xác suất, không phán xét, có gợi ý hỗ trợ tích cực.
        + Đặt là "failed" nếu: Dữ liệu đầu vào không đủ căn cứ, hoặc có nguy cơ suy diễn ngoài dữ liệu.

    NGUYÊN TẮC BẢO VỆ NGHIÊM NGẶT (PEOPLE ANALYTICS GUARDRAILS):
    - TUYỆT ĐỐI KHÔNG suy diễn hoặc tự bịa ra các lý do liên quan đến đời tư, gia đình, mâu thuẫn cá nhân hay tâm lý không xuất hiện trong dữ liệu được cung cấp.
    - TUYỆT ĐỐI KHÔNG viết các câu khẳng định chắc chắn như "nhân viên này chắc chắn sẽ nghỉ việc", "nhân viên đã mất động lực", hay "công ty nên chuẩn bị tuyển người thay thế".
    - Chỉ đưa ra khuyến nghị xoay quanh việc lắng nghe, hỗ trợ và cải thiện môi trường làm việc.
    """


def format_user_prompt(
    risk_score: float,
    risk_band: str,
    top_drivers: List[Dict[str, Any]],
    department: str = "N/A",
    role: str = "N/A"
) -> str:
    drivers_lines = []
    for d in top_drivers:
        feature = d.get("feature", "chỉ số")
        val = d.get("value", "N/A")
        impact = d.get("impact", "")
        drivers_lines.append(f"- {feature}: {val} ({impact})")

    drivers_str = "\n".join(drivers_lines) if drivers_lines else "- Không ghi nhận tín hiệu bất thường rõ rệt"

    return f"""Dữ liệu dự báo rủi ro nhân viên:
        - Phòng ban: {department}
        - Vị trí / Chức danh: {role}
        - Xác suất nghỉ việc trong 90 ngày: {risk_score * 100:.1f}%
        - Nhóm rủi ro (Risk Band): {risk_band}
        - Các tín hiệu rủi ro chính ghi nhận được (Top Drivers):
        {drivers_str}
        Hãy phân tích và xuất dữ liệu có cấu trúc theo đúng định dạng được yêu cầu."""


# ==============================================================================
# 3. LLM Builder
# ==============================================================================
def build_llm(llm_config: Optional[LLMConfig] = None) -> Any:
    """Tạo client Chat LLM dựa trên cấu hình provider."""
    config = llm_config or LLMConfig(
        provider=getattr(settings, "LLM_PROVIDER", "openai"),
        model_name=getattr(settings, "LLM_MODEL", "gpt-4o-mini"),
        api_key=getattr(settings, "OPENAI_API_KEY", "") or None,
        api_url=getattr(settings, "LLM_URL", None)
    )

    if not config.api_key and config.provider != "local":
        return None

    try:
        if config.provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=config.model_name,
                api_key=config.api_key,
                temperature=0.3
            )
        elif config.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=config.model_name,
                api_key=config.api_key,
                temperature=0.3
            )
        elif config.provider == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=config.model_name,
                api_key=config.api_key,
                temperature=0.3
            )
        elif config.provider == "local":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=config.model_name,
                api_key="not-needed",
                base_url=config.api_url,
                temperature=0.3
            )
    except ImportError:
        return None

    return None


# ==============================================================================
# 4. Explanation Service with Structured Output & Fallback
# ==============================================================================
class ExplanationService:
    def __init__(self):
        self.llm_config = LLMConfig(
            provider=getattr(settings, "LLM_PROVIDER", "openai"),
            model_name=getattr(settings, "LLM_MODEL", "gpt-4o-mini"),
            api_key=getattr(settings, "OPENAI_API_KEY", "") or None,
            api_url=getattr(settings, "LLM_URL", None)
        )
        self.llm = build_llm(self.llm_config)

    def _fallback_explanation(
        self,
        risk_score: float,
        risk_band: str,
        top_drivers: List[Dict[str, Any]]
    ) -> ExplainResponse:
        """Tạo phản hồi chuẩn mực có guardrails khi không có API key hoặc network lỗi."""
        driver_desc = "; ".join([d.get("impact", "") for d in top_drivers if d.get("impact")])
        if not driver_desc:
            driver_desc = "các chỉ số vận hành duy trì ở mức bình thường"

        explanation = (
            f"Nhân viên được ghi nhận ở nhóm rủi ro {risk_band} với xác suất dự báo {risk_score * 100:.1f}%. "
            f"Các tín hiệu chính từ dữ liệu quan sát gồm: {driver_desc}. "
            "Đây là chỉ số cảnh báo sớm để hỗ trợ người quản lý chủ động đồng hành và hỗ trợ, "
            "không phải là kết luận chắc chắn về quyết định cá nhân của nhân viên."
        )

        topics = [
            "Rà soát khối lượng công việc và cân bằng thời gian (Work-Life Balance)",
            "Định hướng mục tiêu phát triển và kế hoạch thăng chức dài hạn",
            "Mức độ hỗ trợ và phản hồi định kỳ từ người quản lý trực tiếp"
        ]

        return ExplainResponse(
            explanation_text=explanation,
            suggested_topics=topics,
            guardrail_status="passed"
        )

    def generate_llm_explanation(
        self,
        payload: Optional[ExplainInput | Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Sinh văn bản giải thích cấu trúc cho HR.
        Hỗ trợ gọi Chat LLM với with_structured_output hoặc tự động fallback an toàn.
        """
        if payload is not None:
            if isinstance(payload, ExplainInput):
                risk_score = payload.churn_probability
                risk_band = payload.risk_band
                top_drivers = payload.top_drivers
                department = payload.department or "N/A"
                role = payload.role or "N/A"
            else:
                risk_score = payload.get("churn_probability", 0.0)
                risk_band = payload.get("risk_band", "N/A")
                top_drivers = payload.get("top_drivers", [])
                department = payload.get("department", "N/A")
                role = payload.get("role", "N/A")
        else:
            risk_score = kwargs.get("risk_score", 0.0)
            risk_band = kwargs.get("risk_band", "N/A")
            top_drivers = kwargs.get("top_drivers", [])
            department = kwargs.get("department", "N/A")
            role = kwargs.get("role", "N/A")

        # 1. Thử gọi LLM với Structured Output
        if self.llm is not None:
            try:
                structured_llm = self.llm.with_structured_output(ExplainResponse)
                user_message = format_user_prompt(
                    risk_score=risk_score,
                    risk_band=risk_band,
                    top_drivers=top_drivers,
                    department=department,
                    role=role
                )
                messages = [
                    ("system", HR_REASONING_SYSTEM_PROMPT),
                    ("human", user_message)
                ]
                output: ExplainResponse = structured_llm.invoke(messages)
                return output.model_dump()
            except Exception:
                # Nếu API gặp sự cố (quota, timeout...), chuyển tiếp sang Fallback an toàn
                pass

        # 2. Offline Fallback Generator
        fallback_output = self._fallback_explanation(risk_score, risk_band, top_drivers)
        return fallback_output.model_dump()


# ==============================================================================
# 5. Singleton Accessors
# ==============================================================================
@lru_cache(maxsize=1)
def get_explanation_service() -> ExplanationService:
    return ExplanationService()


def generate_llm_explanation(
    payload: Optional[ExplainInput | Dict[str, Any]] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """Helper cấp module để tương thích trực tiếp với app/main.py."""
    service = get_explanation_service()
    return service.generate_llm_explanation(payload=payload, **kwargs)