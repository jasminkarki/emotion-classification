import os
import pickle

from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.tree import DecisionTreeClassifier

from .config import config


def base_model(X_train_v, X_test_v, y_train, y_test, classifier, alpha):

    mnb = classifier(alpha=alpha)
    mnb.fit(X_train_v, y_train)
    pickle.dump(
        mnb,
        open(
            os.path.join(
                config.BASE_DIR, "emotion_classify", "pickle_files", "model_nb.pkl"
            ),
            "wb",
        ),
    )
    model = pickle.load(
        open(
            os.path.join(
                config.BASE_DIR, "emotion_classify", "pickle_files", "model_nb.pkl"
            ),
            "rb",
        )
    )
    predictions = model.predict(X_test_v)
    f1_sc = metrics.f1_score(y_test, predictions, average="macro")
    accuracy_sc = metrics.accuracy_score(y_test, predictions)
    return f1_sc, accuracy_sc


def inference(text):
    vector = pickle.load(
        open(
            os.path.join(
                config.BASE_DIR, "emotion_classify", "pickle_files", "vector.pkl"
            ),
            "rb",
        )
    )
    vectorized = vector.transform([text])
    model = pickle.load(
        open(
            os.path.join(
                config.BASE_DIR, "emotion_classify", "pickle_files", "model_nb.pkl"
            ),
            "rb",
        )
    )
    label = model.predict(vectorized)
    label_inv_encoder = pickle.load(
        open(
            os.path.join(
                config.BASE_DIR, "emotion_classify", "pickle_files", "label_encoder.pkl"
            ),
            "rb",
        )
    )
    output = label_inv_encoder.inverse_transform(label)
    return output
