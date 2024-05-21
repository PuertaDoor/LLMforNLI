import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time
import psutil

def load_model(model_path):
    # Load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path, device_map='cuda:0')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def predict(model, tokenizer, premise, hypothesis):
    # Utilize the specific task description format for mT5
    task_description = "Determine if the hypothesis is true based on the premise."
    text = f"Task: {task_description} Premise: {premise} Hypothesis: {hypothesis} Label (Entailment, Neutral, Contradiction):"

    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss  # Memory in bytes

    # Tokenize the input text
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=1000
    )

    inputs = inputs.to(model.device)

    # Perform inference
    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits

    elapsed_time = time.time() - start_time
    end_memory = psutil.Process().memory_info().rss
    memory_used = end_memory - start_memory

    print(f"Text length: {len(text)} characters")
    print(f"Inference time: {elapsed_time:.4f} seconds")
    print(f"Memory used: {memory_used / (1024 ** 2):.4f} Mo")  # Convert to MB
    
    return logits

def main():
    model_path = "./best_model_mt5"
    model, tokenizer = load_model(model_path)
    
    while True:
        # Input from user
        premise = input("Enter the premise: ")
        hypothesis = input("Enter the hypothesis: ")
        
        # Prediction
        probs = predict(model, tokenizer, premise, hypothesis)
        pred_label = torch.argmax(probs, dim=1)
        
        # Mapping indices to labels
        labels = ['contradiction', 'entailment', 'neutral']
        print(f"Prediction: {labels[pred_label]}")
        if input("Try another (yes/no)? ").lower() != 'yes':
            break

if __name__ == "__main__":
    main()
