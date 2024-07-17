import pickle
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def load_model(model_name):
    """Load a pre-trained model and tokenizer from Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def serialize_model(model):
    """Serialize the model state_dict."""
    buffer = pickle.dumps(model.state_dict())
    return buffer

def serialize_tokenizer(tokenizer, temp_dir="./temp_tokenizer"):
    """Serialize the tokenizer."""
    tokenizer.save_pretrained(temp_dir)
    with open(f"{temp_dir}/tokenizer_config.json", 'rb') as f:
        tokenizer_data = f.read()
    return tokenizer_data

def deserialize_model(serialized_model, model_name):
    """Deserialize the model state_dict."""
    state_dict = pickle.loads(serialized_model)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.load_state_dict(state_dict)
    return model

def deserialize_tokenizer(serialized_tokenizer, temp_dir="./temp_tokenizer"):
    """Deserialize the tokenizer."""
    with open(f"{temp_dir}/tokenizer_config.json", 'wb') as f:
        f.write(serialized_tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(temp_dir)
    return tokenizer

def fine_tune_model(model, tokenizer, train_data, output_dir, epochs=3):
    """Fine-tune the model using the provided training data."""
    train_dataset = load_dataset('text', data_files={'train': train_data})['train']

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    tokenized_dataset = train_dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def generate_text(model, tokenizer, prompt, max_length=50):
    """Generate text using the fine-tuned model."""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)

def load_model(model_name):
    """Load a pre-trained model and tokenizer from Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def fine_tune_model(model, tokenizer, train_data, output_dir, epochs=3):
    """Fine-tune the model using the provided training data."""
    train_dataset = load_dataset('text', data_files={'train': train_data})['train']

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    tokenized_dataset = train_dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logging.info(f"Model fine-tuned and saved at {output_dir}")

def generate_text(model, tokenizer, prompt, max_length=50):
    """Generate text using the fine-tuned model."""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
