import os
import re
import pickle
import argparse
import sys
from loguru import logger

import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from emotion_classify.config.config import *

# Configure loguru logger
logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")

def load_dataset(dataset_path: str, dataset_name: str) -> pd.DataFrame:
    """Load dataset from the given path and encode labels."""
    df = pd.read_csv(
        os.path.join(dataset_path, dataset_name),
        names=["ind", "emotion", "text"],
        header=None,
    )
    df.drop(columns=["ind"], inplace=True)
    labels = df["emotion"].values
    label_encoder = LabelEncoder().fit(labels)
    
    label_encoder_path = os.path.join(CHECKPOINT_DIR, "label_encoder.pkl")
    with open(label_encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)    
    logger.info("Label encoder saved at {}", label_encoder_path)
    
    df["emotion"] = label_encoder.transform(labels)
    return df

def preprocess_text(text: str) -> str:
    """Clean and preprocess a single text entry."""
    text = text.lower()
    text = re.sub(r"\W+", " ", text)
    text = re.sub(r"\d", "", text)
    text = re.sub(r"[\t\n]+", "", text)
    text = re.sub(r"[#,@,&,!]", " ", text)
    text = re.sub(r"\s+", " ", text)
    
    stop_words = set(stopwords.words("english"))
    words = [word for word in word_tokenize(text) if word not in stop_words]
    return " ".join(words)

def preprocessing(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply text preprocessing to the dataset."""
    dataframe["text"] = dataframe["text"].apply(preprocess_text)
    return dataframe

def train_test(dataframe: pd.DataFrame):
    """Split the dataset into training and testing sets."""
    return train_test_split(
        dataframe["text"], dataframe["emotion"], random_state=RANDOM_STATE, test_size=TEST_SIZE
    )


def feature_vectorizer(X_train, X_test, vectorizer):

    vector = vectorizer(stop_words="english", lowercase=False)
    vector.fit(X_train)
    vector_path = os.path.join(CHECKPOINT_DIR, "vector.pkl")
    pickle.dump(vector, open(vector_path, "wb"))
    logger.info("Feature vectorizer saved at {}", vector_path)
    
    return vector.transform(X_train), vector.transform(X_test)