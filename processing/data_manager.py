import joblib
from config.config import TRAINED_MODEL_DIR, MODEL_NAME

def load_model():
  model_path = TRAINED_MODEL_DIR / MODEL_NAME
  model = joblib.load(model_path)
  return model

def load_feature_names():
  feature_names_path = TRAINED_MODEL_DIR / "feature_names.pkl"
  features_names = joblib.load(features_name_path)
  return features_names 


def save_model(model, filename: str):
  """Save a trained model to disk."""
  model_path = TRAINED_MODEL_DIR / filename
  joblib.dump(model, model_path)
  print(f"Model saved to {model_path}")