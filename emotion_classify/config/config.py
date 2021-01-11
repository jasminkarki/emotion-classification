import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = os.path.join(BASE_DIR, "data")

RAW_DATAPATH = os.path.join(DATA_PATH, "raw")

DATASET_NAME = "ISEAR.csv"

shared_components = {"db": None}
