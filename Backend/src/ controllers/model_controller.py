from flask import request, jsonify
from src.services.model_service import (
    load_model, fine_tune_model, generate_text, serialize_model, serialize_tokenizer,
    deserialize_model, deserialize_tokenizer
)
from src.services.mongodb_service.py import store_model_in_db, load_model_from_db

model_name = "distilgpt2"
model, tokenizer = load_model(model_name)

def fine_tune():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    output_dir = "fine_tuned_model"
    fine_tune_model(model, tokenizer, file, output_dir)

    serialized_model = serialize_model(model)
    serialized_tokenizer = serialize_tokenizer(tokenizer)
    store_model_in_db(model_name, serialized_model, serialized_tokenizer)

    return jsonify({"status": "Model fine-tuned and stored in MongoDB"}), 200

def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    serialized_model, serialized_tokenizer = load_model_from_db(model_name)
    model = deserialize_model(serialized_model, model_name)
    tokenizer = deserialize_tokenizer(serialized_tokenizer)

    response = generate_text(model, tokenizer, prompt)
    return jsonify({"response": response}), 200
