from flask import Flask, Response
from flask_cors import CORS
from flask import request, jsonify
from bson.json_util import dumps
from flask_pymongo import PyMongo
from config.config import shared_components
import numpy as np
import pickle
import json
import ast
from emotion_classify import baseline_model

app = Flask(__name__)  

model = pickle.load(open('./emotion_classify/model_nb.pkl','rb'))
app.config.from_pyfile('config/config.cfg')
CORS(app)
mongo = PyMongo(app)

# Select the database
db = mongo.db
shared_components['db'] = db

def create_log(params):
    """
    Function to create new logs.
    """
    db = shared_components["db"]
    collection = db.log   # Collection name
    try:
        # Insert Text
        collection.insert(json.loads(params))

    except Exception as e:
        # Error while trying to insert the text
        print("Exception: {}".format(e))

@app.route('/api/logs',methods=['GET']) 
def fetch_logs():
    """
       Function to fetch the users.
    """
    db = shared_components["db"]
    collection = db.log
    try:
        # Fetch all the texts
        logs_fetched = collection.find({},{'_id':0})

        # Check if the texts are found
        if logs_fetched.count() > 0:
            # Prepare the response
            records = dumps(logs_fetched)
            resp = Response(records, status=200, mimetype='application/json')
            return resp
        else:
            # No records are found
            return Response("No records are found", status=404)
    except Exception as e:
        print("Exception: {}".format(e))
        # Error while trying to fetch the resource
        return Response("Error while trying to fetch the resource", status=500)

@app.route('/api/log/<log_id>',methods=['GET']) 
def fetch_log(log_id):
    """
       Function to fetch the log.
    """
    db = shared_components["db"]
    collection = db.log
    try:
        # Fetch one the record(s)
        records_fetched = collection.find_one({"id": log_id},{'_id':0})

        # Check if the records are found
        if records_fetched:
            # Prepare the response
            records = dumps(records_fetched)
            resp = Response(records, status=200, mimetype='application/json')
            return resp
        else:
            # No records are found
            return Response("No records are found", status=404)
    except Exception as e:
        print("Exception: {}".format(e))
        # Error while trying to fetch the resource
        return Response("Error while trying to fetch the resource", status=500)

@app.route('/api/log/<log_id>',methods=['UPDATE']) 
def update_log(log_id):
    """
       Function to update the user.
    """

    db = shared_components["db"]
    collection = db.user
    try:
        # Get the value which needs to be updated
        if request.json:
            body = ast.literal_eval(json.dumps(request.json))
        else:
            raise TypeError("Invalid request format")

        # Updating the user
        records_updated = collection.update_one(
            {"id": log_id}, {"$set": body})

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
        print("Exception: {}".format(e))
        return Response("Error while updating the resource", status=500)

@app.route('/api/logs/<log_id>',methods=['DELETE']) 
def remove_user(log_id):
    """
       Function to remove the user.
    """
    db = shared_components["db"]
    collection = db.user
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
        print("Exception: {}".format(e))
        return Response("Resource deletion failed", status=500)


@app.route('/')   
def display_page():
    return "This is the landing page"
    # return render_template("home.html")

@app.route('/api/predict',methods=['POST'])   
def predict():
    text_inp = request.get_data()
    data= json.loads(text_inp)
    # # Send for prediction
    output = baseline_model.inference(data["data"])
    if output:
        output=output[0]
    create_log({"output":output, "text": data["data"]})
    return jsonify({"output": str(output)})



if __name__== "__main__":
    app.run("0.0.0.0", 3000, debug=True)   # Automatic run of script