from fastapi import APIRouter
from app.schemas import CrashInput, CrashOutput
from processing.features import prepare_input
from processing.data_manager import load_model

api_router = APIRouter()
model = load_model()


@api_router.get("/health")
def health():
    return {"status": "ok"}


@api_router.post("/predict", response_model=CrashOutput)
def predict(input_data: CrashInput):
    df = prepare_input(input_data.dict())
    crash_prob = model.predict_proba(df)[0][1]
    crash_pred = int(crash_prob >= 0.5)

    if crash_prob < 0.3:
        risk_level = "LOW"
    elif crash_prob < 0.6:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    return CrashOutput(
        crash_probability=round(float(crash_prob), 4),
        crash_predicted=crash_pred,
        risk_level=risk_level
    )