import os
import numpy as np
import pandas as pd

import re
import pickle

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from .config import config


def load_dataset(dataset_path, dataset_name):

    df = pd.read_csv(
        os.path.join(dataset_path, dataset_name),
        names=["ind", "emotion", "text"],
        header=None,
    )

    df.drop(columns=["ind"], inplace=True)
    labels = df["emotion"].values
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(labels)
    pickle.dump(
        label_encoder,
        open(
            os.path.join(
                config.BASE_DIR, "emotion_classify", "pickle_files", "label_encoder.pkl"
            ),
            "wb",
        ),
    )
    df["emotion"] = label_encoder.transform(labels)
    return df


def preprocessing(df):

    df["text"] = df["text"].str.replace("\W+", " ", regex=True)
    df["text"] = df["text"].apply(lambda x: " ".join(x.lower() for x in x.split()))
    df["text"] = df["text"].str.replace("\d", "")
    df["text"] = df["text"].str.replace(r"[\t\n]+", "", regex=True)
    df["text"] = df["text"].str.replace("[#,@,&,!]", " ")
    df["text"] = df["text"].str.replace("\s+", " ", regex=True)
    stop_words = stopwords.words("english")
    df["text"] = df["text"].apply(
        lambda x: " ".join([word for word in x.split() if word not in (stop_words)])
    )
    return df


def train_test(df):

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["emotion"], random_state=64, test_size=0.2
    )
    return X_train, X_test, y_train, y_test


def feature_vectorizer(X_train, X_test, vectorizer):

    vector = vectorizer(stop_words="english", lowercase=False)
    vector.fit(X_train)
    pickle.dump(
        vector,
        open(
            os.path.join(
                config.BASE_DIR, "emotion_classify", "pickle_files", "vector.pkl"
            ),
            "wb",
        ),
    )
    X_train = vector.transform(X_train)
    X_test = vector.transform(X_test)
    return X_train, X_test
