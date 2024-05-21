import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
import time
import psutil

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

def load_model(model_path):
    torch.cuda.empty_cache()
    # Load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path, quantization_config=bnb_config, num_labels=3, device_map={'':torch.cuda.current_device()})
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=True)
    return model, tokenizer

def predict(model, tokenizer, premise, hypothesis):
    torch.cuda.empty_cache()

    text = f"Is this true? {premise} implies {hypothesis}"

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
    model_path = "./best_model_llama"
    model, tokenizer = load_model(model_path)

    model.config.pad_token_id = model.config.eos_token_id

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
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
