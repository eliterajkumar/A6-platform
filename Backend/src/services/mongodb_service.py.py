
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


import os
import bson
from pymongo import MongoClient

# MongoDB connection
def get_database():
    client = MongoClient(os.getenv("MONGO_DB_CONNECTION_STRING"))
    db = client[os.getenv("DB_NAME")]
    return db

db = get_database()
collection = db['models']

def store_model_in_db(model_name, serialized_model, serialized_tokenizer):
    """Store the serialized model and tokenizer in MongoDB."""
    model_data = {
        "model_name": model_name,
        "model": bson.Binary(serialized_model),
        "tokenizer": bson.Binary(serialized_tokenizer)
    }
    collection.insert_one(model_data)
    print("Model and tokenizer stored in MongoDB")

def load_model_from_db(model_name):
    """Load the serialized model and tokenizer from MongoDB."""
    stored_model_data = collection.find_one({"model_name": model_name})
    return stored_model_data['model'], stored_model_data['tokenizer']
