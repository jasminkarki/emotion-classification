import ast
import json
import pickle
import random
import os

import numpy as np
from bson.json_util import dumps
from flask import Flask, send_from_directory, current_app, Response, jsonify, request, render_template, url_for, redirect
from flask_cors import CORS
from flask_pymongo import PyMongo
from emotion_classify.baseline_model import inference, explain_prediction
from emotion_classify.config.config import shared_components, BASE_DIR, CHECKPOINT_DIR
from services.log import (
    home,
    fetch_log,
    fetch_logs,
    predict,
    remove_log,
    update_log,
)

def init_app():
    app = Flask(__name__, static_folder='static')
    app = create_route(app)
    app.config['MIME_TYPES'] = {'html': 'text/html'}

    model = pickle.load(
        open(os.path.join(CHECKPOINT_DIR, "model_nb.pkl"), "rb")
    )
    app.config.from_pyfile(os.path.join(BASE_DIR, "emotion_classify/config/config.cfg"))
    CORS(app)
    mongo = PyMongo(app)

    # Select the database
    db = mongo.db
    shared_components["db"] = db
    return app

def serve_lime_explanation():
    try:
        return send_from_directory('static', 'lime_explanation.html')
    except FileNotFoundError:
        current_app.logger.error("LIME explanation file not found")
        return "LIME explanation file not found", 404

def home():
    return render_template('index.html')

def predict():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    predicted_emotion, probabilities, word_contributions = explain_prediction(text)

    # Prepare data for MongoDB storage
    prediction_data = {
        "text": text,
        "predicted_emotion": predicted_emotion,
        "probabilities": probabilities,
        "word_contributions": word_contributions
    }

    # Insert into MongoDB
    db = shared_components["db"]
    db.log.insert_one(prediction_data)

    return jsonify(prediction_data)


def create_route(app):
    """
    Adds different rules to the urls

    Examples:
        1.  app.add_url_rule(rule="/", view_func=display_page, methods=["GET"])
            URL: / (Homepage), Function: display_page,  Method: GET
        Equivalent code using @app.route decorator:
            @app.route("/", methods=["GET"])

        2.  app.add_url_rule(rule="/api/predict", view_func=predict, methods=["POST"])
            URL: /api/predict, Function: predict,  Method: POST
        Equivalent code using @app.route decorator:
            @app.route("/api/predict", methods=["POST"])
    """
    app.add_url_rule(rule="/", view_func=home, methods=["GET"])
    app.add_url_rule(rule="/api/predict", view_func=predict, methods=["POST"])
    app.add_url_rule(rule="/api/logs", view_func=fetch_logs, methods=["GET"])
    app.add_url_rule(rule="/api/log/<log_id>", view_func=fetch_log, methods=["GET"])
    app.add_url_rule(rule="/api/log/<log_id>", view_func=update_log, methods=["UPDATE"])
    app.add_url_rule(rule="/api/log/<log_id>", view_func=remove_log, methods=["DELETE"])
    app.add_url_rule(rule="/lime_explanation", view_func=serve_lime_explanation, methods=["GET"])
    
    return app


app = init_app()

if __name__ == "__main__":
    app.run("0.0.0.0", 3000, debug=True)  # Automatic run of script

