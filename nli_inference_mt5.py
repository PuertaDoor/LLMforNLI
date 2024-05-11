import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model(model_path):
    # Load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def predict(model, tokenizer, premise, hypothesis):
    # Utilize the specific task description format for mT5
    task_description = "Determine if the hypothesis is true based on the premise."
    text = f"Task: {task_description} Premise: {premise} Hypothesis: {hypothesis}"

    # Tokenize the input text
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=1000
    )

    # Perform inference
    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits

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
