from pydantic import BaseModel


class CrashInput(BaseModel):
    SP500_Return_5d: float
    SP500_Return_21d: float
    SP500_Return_63d: float
    VIX: float
    Gold_Return_63d: float
    Gold_Return_21d: float
    Gold_Return_5d: float
    Oil_Return_63d: float
    Oil_Return_21d: float
    Oil_Return_5d: float
    T10Y: float
    CPI: float
    Unemployment: float
    Yield_Spread: float
    GDP_Growth: float
    Gov_Expenditure: float
    Credit_Spread: float
    CAPE: float
    Sahm_Rule: float
    Oil_Price_Ratio: float
    Gold_Price_Ratio: float
    SP500_Price_Ratio: float


class CrashOutput(BaseModel):
    crash_probability: float
    crash_predicted: int
    risk_level: str
