import ast
import json
import pickle
import random
import os

import numpy as np
from bson.json_util import dumps
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from flask_pymongo import PyMongo

from emotion_classify.config.config import shared_components, BASE_DIR
from services.log import (
    display_page,
    fetch_log,
    fetch_logs,
    predict,
    remove_log,
    update_log,
)


def init_app():
    app = Flask(__name__)
    app = create_route(app)

    model = pickle.load(
        open(os.path.join(BASE_DIR, "emotion_classify/checkpoints/model_nb.pkl"), "rb")
    )
    app.config.from_pyfile(os.path.join(BASE_DIR, "emotion_classify/config/config.cfg"))
    CORS(app)
    mongo = PyMongo(app)

    # Select the database
    db = mongo.db
    shared_components["db"] = db
    return app


def create_route(app):
    """
    Adds different rules to the urls
    """
    app.add_url_rule(rule="/", view_func=display_page, methods=["GET"])
    app.add_url_rule(rule="/api/predict", view_func=predict, methods=["POST"])
    app.add_url_rule(rule="/api/logs", view_func=fetch_logs, methods=["GET"])
    app.add_url_rule(rule="/api/log/<log_id>", view_func=fetch_log, methods=["GET"])
    app.add_url_rule("/api/log/<log_id>", view_func=update_log, methods=["UPDATE"])
    app.add_url_rule("/api/log/<log_id>", view_func=remove_log, methods=["DELETE"])
    return app


app = init_app()

if __name__ == "__main__":
    app.run("0.0.0.0", 3000, debug=True)  # Automatic run of script

