import ast
import json
import os
import pickle
import random
import logging
logging.basicConfig(level=logging.ERROR)

from bson.json_util import dumps
from emotion_classify.baseline_model import explain_prediction
from emotion_classify.config.config import shared_components
from flask import Response, jsonify, request


def get_collection(collection_name):
    db = shared_components["db"]
    return db[collection_name]

def create_log(params):
    """
    Function to create new logs.
    """
    collection = get_collection("log")  
    try:
        collection.insert_one(params)  # Directly insert dictionary
    except Exception as e:
        logging.error(f"Exception: {format(e)}")


def fetch_logs():
    """
       Function to fetch the users.
    """
    collection = get_collection("log")
    try:
        # Fetch all the texts
        logs_fetched = collection.find({}, {"_id": 0})
        print(dumps(logs_fetched))
        # Check if the texts are found
        if logs_fetched:
            # Prepare the response
            return Response(dumps(logs_fetched), status=200, mimetype="application/json")
        else:
            # No records are found
            return Response("No records are found", status=404)
    except Exception as e:
        logging.error(f"Exception: {format(e)}")
        # Error while trying to fetch the resource
        return Response("Error while trying to fetch the resource", status=500)


def fetch_log(log_id):
    """
       Function to fetch the log.
    """
    collection = get_collection("log")
    try:
        # Fetch one the record(s)
        records_fetched = collection.find_one({"id": log_id}, {"_id": 0})
        # records_fetched = list(collection.find({}, {"_id": 0}))  # Convert to list

        # Check if the records are found
        if records_fetched:
            # Prepare the response
            return Response(dumps(records_fetched), status=200, mimetype="application/json")
        else:
            # No records are found
            return Response("No records are found", status=404)
    except Exception as e:
        logging.error(f"Exception: {format(e)}")
        # Error while trying to fetch the resource
        return Response("Error while trying to fetch the resource", status=500)


def update_log(log_id):
    """
       Function to update the user.
    """
    collection = get_collection("log")
    try:
        # Get the value which needs to be updated
        if request.json:
            body = ast.literal_eval(json.dumps(request.json))
        else:
            raise TypeError("Invalid request format")

        # Updating the user
        records_updated = collection.update_one({"id": log_id}, {"$set": body})

        # Check if resource is updated
        if records_updated.modified_count > 0:
            # Prepare the response as resource is updated successfully
            return Response("Resource updated successfully", status=200)
        else:
            # Bad request as the resource is not available to update
            # Add message for debugging purpose
            return Response("Resource not available", status=404)
    except Exception as e:
        # Error while trying to update the resource
        # Add message for debugging purpose
        logging.error(f"Exception: {format(e)}")
        return Response("Error while updating the resource", status=500)


def remove_log(log_id):
    """
       Function to remove the user.
    """
    collection = get_collection("log")
    try:
        # Delete the user
        delete_user = collection.delete_one({"id": log_id})

        if delete_user.deleted_count > 0:
            # Prepare the response
            return Response("Resource deleted successfully", status=200)
        else:
            # Resource Not found
            return Response("Resource Not Found", status=404)
    except Exception as e:
        # Error while trying to delete the resource
        # Add message for debugging purpose
        logging.error(f"Exception: {format(e)}")
        return Response("Resource deletion failed", status=500)


# def fetch_predictions():
#     get_collection("predictions")
#     predictions = list(collection.find({}, {"_id": 0}))  # Exclude MongoDB's ObjectId
#     return jsonify(predictions)


def home():
    """Welcome message for the API."""
    # Message to the user
    message = {
        "api_version": "v1.0",
        "status": "200",
        "message": "Welcome to the Flask API",
    }
    # Making the message looks good
    resp = jsonify(message)

    # Returning the object
    return resp

def predict():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Get predictions
    emotion, probabilities, explanation = explain_prediction(text)

    # Create a structured log entry
    result = {
        "text": text,
        "emotion": emotion,
        "probabilities": probabilities,
        "explanation": explanation,
    }

    # Store in MongoDB
    create_log(json.dumps(result))  

    return jsonify(result)  # Return the JSON response

# def predict():
#     text_inp = request.get_data()
#     data = json.loads(text_inp)
#     # # Send for prediction
#     output = baseline_model.inference(data["data"])
#     if output:
#         output = output[0]
#     create_log(
#         json.dumps(
#             {"output": output, "text": data["data"], "id": str(random.getrandbits(8))}
#         )
#     )
#     return jsonify({"output": str(output)})
