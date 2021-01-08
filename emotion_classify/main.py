import os
import pickle
import sys

import mlflow
import numpy as np
import pandas as pd
from mlflow import log_artifacts, log_metric, log_param
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from baseline_model import base_model
from config import config  # insert . to run from outside
from preprocessing import feature_vectorizer, load_dataset, preprocessing, train_test

# Load dataset from its path
dataset = load_dataset(config.RAW_DATAPATH, config.DATASET_NAME)
# Clean the data
dataset = preprocessing(dataset)
# Train test split the loaded dataset
X_train, X_test, y_train, y_test = train_test(dataset)
# Obtain feature vectors of the dataset
X_train, X_test = feature_vectorizer(X_train, X_test, TfidfVectorizer)

alpha = float(sys.argv[1])

with mlflow.start_run():
    f1_score, accuracy = base_model(
        X_train, X_test, y_train, y_test, MultinomialNB, alpha
    )

    print("Multinomial Naive Bayes model")
    print("  f1_score: %s" % f1_score)
    print("  accuracy: %s" % accuracy)

    mlflow.log_param("alpha ", alpha)
    mlflow.log_metric("f1_score", f1_score)
    mlflow.log_metric("accuracy", accuracy)
