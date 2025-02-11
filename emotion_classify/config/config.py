import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PACKAGE_DIR = os.path.join(BASE_DIR, "emotion_classify")

CHECKPOINT_DIR = os.path.join(PACKAGE_DIR, "checkpoints")

DATA_PATH = os.path.join(BASE_DIR, "data")

RAW_DATAPATH = os.path.join(DATA_PATH, "raw")

DATASET_NAME = "ISEAR.csv"
RANDOM_STATE = 64
TEST_SIZE = 0.2

shared_components = {"db": None}
