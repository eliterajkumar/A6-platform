import os
from pymongo import MongoClient

def get_database():
    CONNECTION_STRING = os.getenv("MONGO_DB_CONNECTION_STRING")
    client = MongoClient(CONNECTION_STRING)
    return client[os.getenv("DB_NAME")]

db = get_database()
