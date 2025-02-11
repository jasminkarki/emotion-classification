import os
import pickle
import numpy as np

from lime.lime_text import LimeTextExplainer
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.tree import DecisionTreeClassifier

from .config import config

def base_model(X_train_v, X_test_v, y_train, y_test, classifier, alpha):

    mnb = classifier(alpha=alpha)
    mnb.fit(X_train_v, y_train)
    pickle.dump(mnb, open(os.path.join(config.BASE_DIR, "emotion_classify", "checkpoints", "model_nb.pkl"),"wb"))
    model = pickle.load(open(os.path.join(config.BASE_DIR, "emotion_classify", "checkpoints", "model_nb.pkl"),"rb"))
    predictions = model.predict(X_test_v)
    f1_sc = metrics.f1_score(y_test, predictions, average="macro")
    accuracy_sc = metrics.accuracy_score(y_test, predictions)
    return f1_sc, accuracy_sc

def load_pickle(file_name):
    with open(os.path.join(config.CHECKPOINT_DIR, file_name), "rb") as f:
        return pickle.load(f)

def inference(text):
    vector = load_pickle("vector.pkl")
    model = load_pickle("model_nb.pkl")
    label_inv_encoder = load_pickle("label_encoder.pkl")

    vectorized_text = vector.transform([text])
    probabilities = model.predict_proba(vectorized_text)[0]
    label = model.predict(vectorized_text)
    emotion = label_inv_encoder.inverse_transform(label)
    return emotion, probabilities

def explain_prediction(text):
    vectorizer = load_pickle("vector.pkl")
    model = load_pickle("model_nb.pkl")
    label_inv_encoder = load_pickle("label_encoder.pkl")
    
    class_names = label_inv_encoder.classes_
    explainer = LimeTextExplainer(class_names=class_names)

    def predictor(text_list):
        vectorized_text = vectorizer.transform(text_list)
        return model.predict_proba(vectorized_text)

    explanation = explainer.explain_instance(
        text, predictor, num_features=10, top_labels=5)
    
    # Get predicted probabilities
    vectorized_text = vectorizer.transform([text])
    probabilities = model.predict_proba(vectorized_text)[0]  # Get probabilities for each class
    predicted_label = model.predict(vectorized_text)[0]
    predicted_emotion = label_inv_encoder.inverse_transform([predicted_label])[0]


    # Get word importance using as_list()
    word_contributions = explanation.as_list()

    return predicted_emotion, dict(zip(class_names, probabilities)), word_contributions