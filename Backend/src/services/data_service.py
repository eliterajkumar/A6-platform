from src.utils.data_cleaning import clean_data
from src.utils.tokenization import tokenize_data

def upload_and_process_data(file):
    raw_data = file.read()
    cleaned_data = clean_data(raw_data)
    tokens = tokenize_data(cleaned_data)
    # Store data in MongoDB
    # Return a response with the status
    return {"status": "success"}
