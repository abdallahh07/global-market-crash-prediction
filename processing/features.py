import pandas as pd
from config.config import FEATURES

def prepare_input(input_data:dict) -> pd.DataFrame:
  df = pd.DataFrame([input_data])
  df = df[FEATURES] 
  return df
