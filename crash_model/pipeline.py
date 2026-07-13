from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def create_pipeline(scale_pos_weight:float)-> LGBMClassifier:
  """Create the LightGBM model with best parameters from tuning."""  
  model = LGBMClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1,
        subsample=0.6,
        n_estimators=300,
        min_child_samples=10,
        max_depth=5,
        learning_rate=0.05,
        colsample_bytree=1.0)
  return model