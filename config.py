"""Central configuration: paths, feature groups, color palette, constants."""

DATA_PATH  = "data/telco_churn.csv"
MODEL_PATH = "data/model.pkl"
SHAP_PATH  = "data/shap_values.pkl"

DEMOGRAPHIC_FEATURES = ["gender", "SeniorCitizen", "Partner", "Dependents"]

SERVICE_FEATURES = [
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

ACCOUNT_FEATURES = [
    "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "tenure",
]

CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

MONTHLY_REVENUE = 65.0

BG           = "#0a0a1a"
CARD_BG      = "rgba(18,18,42,0.95)"
BORDER       = "rgba(255,255,255,0.08)"
TEXT         = "#e2e2f0"
MUTED        = "#6b7280"
ACCENT       = "#7c3aed"
GREEN        = "#22c55e"
RED          = "#ef4444"
YELLOW       = "#f59e0b"
BLUE         = "#3b82f6"
ORANGE       = "#f97316"
CHURN_COLOR  = "#ef4444"
RETAIN_COLOR = "#22c55e"
