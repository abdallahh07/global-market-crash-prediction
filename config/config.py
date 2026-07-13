from pathlib import Path
import yaml

ROOT = Path(__file__).parent.parent

CONFIG_PATH = ROOT/"config"/"config.yml"
TRAINED_MODEL_DIR = ROOT/"trained_model"
DATASET_DIR = ROOT/"data"

with open(CONFIG_PATH,"r") as f:
  config = yaml.safe_load(f)
  
MODEL_NAME = config["model"]["pipeline_save_file"]
FEATURES = config["features"]
TARGET = config["target"]
DROP_FEATURES = config["drop_features"]

MODEL_PATH = TRAINED_MODEL_DIR/MODEL_NAME