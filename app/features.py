from __future__ import annotations

TARGET_COLUMN = "label"
ID_COLUMN = "employee_id"


RAW_FEATURES = [
    'employee_id', 'snapshot_date', 'tasks_completed', 'late_tasks',
    'overtime_hours', 'meeting_hours', 'focus_hours',
    'after_hours_messages', 'absent_days', 'performance_score',
    'goal_completion_rate', 'manager_rating', 'peer_feedback_score',
    'review_sentiment_score', 'leave_days', 'manager_support_score',
    'career_growth_score', 'compensation_satisfaction_score', 'leave_type',
    'is_unplanned', 'approved_by_manager', 'department', 'job_level',
    'label', 'role', 'location', 'employment_type', 'hire_date',
    'manager_id', 'salary_band_min_usd', 'salary_band_max_usd',
    'monthly_salary_usd', 'promotion_count', 'last_promotion_date',
    'survey_date', 'engagement_score', 'work_life_balance_score',
    'leave_date', 'review_date', 'week_start'
]

FINAL_FEATURES = [
    # Numeric features (25)
    "tasks_completed", "late_tasks", "overtime_hours", "meeting_hours", "focus_hours",
    "after_hours_messages", "absent_days", "performance_score", "goal_completion_rate",
    "manager_rating", "peer_feedback_score", "review_sentiment_score", "leave_days",
    "manager_support_score", "career_growth_score", "compensation_satisfaction_score",
    "job_level", "salary_band_min_usd", "salary_band_max_usd", "monthly_salary_usd",
    "promotion_count", "engagement_score", "work_life_balance_score", "tenure_months",
    "days_since_last_promotion",
    # Categorical features (7)
    "leave_type", "is_unplanned", "approved_by_manager",
    "department", "role", "location", "employment_type"
]

NUMERIC_FEATURES = FINAL_FEATURES[:25]

CATEGORICAL_FEATURES = FINAL_FEATURES[25:]

DOWN_SAMPLE_STRATIFY_FEATURES = CATEGORICAL_FEATURES

OUTPUT_FEATURES = [
    ID_COLUMN,
    *FINAL_FEATURES,
    TARGET_COLUMN
]