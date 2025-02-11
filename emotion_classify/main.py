import os
import sys
import argparse
from loguru import logger

import nltk
import mlflow
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from emotion_classify.baseline_model import base_model
from emotion_classify.config import config
from emotion_classify.preprocessing import (
    feature_vectorizer, load_dataset, preprocessing, train_test
)

# Configure loguru logger
logger.remove()  # Remove existing handlers to prevent duplicate logs
logger.add(sys.stderr, format="<level>{time} {level} {message}</level>", level="INFO", colorize=True)

# Ensure necessary NLTK resources are available
def download_nltk_resources():
    resources = ["stopwords", "punkt", "punkt_tab"]
    for resource in resources:
        try:
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            nltk.download(resource)

# Run NLTK setup
download_nltk_resources()

def main(alpha: float):
    """Main function to train and evaluate the model."""
    logger.info("Loading dataset...")
    dataset = load_dataset(config.RAW_DATAPATH, config.DATASET_NAME)
    dataset = preprocessing(dataset)
    X_train, X_test, y_train, y_test = train_test(dataset)
    X_train, X_test = feature_vectorizer(X_train, X_test, TfidfVectorizer)

    logger.info("Starting model training...")
    with mlflow.start_run():
        f1_score, accuracy = base_model(X_train, X_test, y_train, y_test, MultinomialNB, alpha)

        logger.success("Multinomial Naive Bayes model results:")
        logger.info("  f1_score: {}", f1_score)
        logger.info("  accuracy: {}", accuracy)

        mlflow.log_param("alpha", alpha)
        mlflow.log_metric("f1_score", f1_score)
        mlflow.log_metric("accuracy", accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an emotion classification model.")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha value for Naive Bayes")
    args = parser.parse_args()
    
    try:
        main(args.alpha)
    except Exception as e:
        logger.error("An error occurred: {}", str(e))
        sys.exit(1)