import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import T5ForConditionalGeneration, T5Tokenizer, MarianMTModel, MarianTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Load datasets
datasets = {
    "FEVER": "pietrolesci/nli_fever",
    "CLIMATE-FEVER": "Jasontth/climate_fever_plus",
    "XFACT": "metaeval/x-fact",
    "SciTail": ("allenai/scitail", "tsv_format"),
    "VitaminC": "tals/vitaminc"
}

def load_datasets(datasets):
    data = []
    print("Loading datasets...")
    for name, path in tqdm(datasets.items()):
        if isinstance(path, tuple):
            dataset = load_dataset(path[0], path[1])
        else:
            dataset = load_dataset(path)
        if "train" in dataset:
            data.append(dataset["train"].to_pandas())
        elif "validation" in dataset:
            data.append(dataset["validation"].to_pandas())
        elif "test" in dataset:
            data.append(dataset["test"].to_pandas())
    print("Datasets loaded.")
    return pd.concat(data)

evidence_data = load_datasets(datasets)

# Ensure all entries in 'evidence_texts' are valid strings
print("Preparing evidence texts...")
evidence_texts = [str(evidence) for evidence in evidence_data['evidence'] if isinstance(evidence, str)]
print("Evidence texts prepared.")

# Prepare the semantic similarity model
sts_model_name = "Gameselo/STS-multilingual-mpnet-base-v2"
print("Loading semantic similarity model...")
sts_model = SentenceTransformer(sts_model_name)
print("Semantic similarity model loaded.")

# Prepare the NLI model FLAN-T5-XXL
flan_t5_model_name = "google/flan-t5-xxl"
print("Loading NLI model...")
flan_t5_tokenizer = T5Tokenizer.from_pretrained(flan_t5_model_name)
flan_t5_model = T5ForConditionalGeneration.from_pretrained(flan_t5_model_name)
print("NLI model loaded.")

# Prepare the translation model for any language to English
translation_model_name = "Helsinki-NLP/opus-mt-mul-en"
print("Loading translation model...")
translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
translation_model = MarianMTModel.from_pretrained(translation_model_name)
print("Translation model loaded.")

# Function to translate text to English if necessary
def translate_to_english(text):
    print("Translating input to English...")
    inputs = translation_tokenizer(text, return_tensors="pt", padding=True)
    translated = translation_model.generate(**inputs)
    translated_text = translation_tokenizer.decode(translated[0], skip_special_tokens=True)
    print("Translation complete.")
    return translated_text

# Function to perform NLI
def perform_nli(premise, hypothesis):
    prompt = f"Task: Determine if the hypothesis is true based on the premise. Premise: \"{premise}\" Hypothesis: \"{hypothesis}\" Label (Entailment, Neutral, Contradiction):"
    inputs = flan_t5_tokenizer(prompt, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt")
    print("Performing NLI...")
    outputs = flan_t5_model.generate(**inputs)
    label = flan_t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("NLI complete.")
    return label

# Main function to process user input
def infer_user_input(user_input, evidence_texts):
    translated_input = translate_to_english(user_input)
    print("Encoding user input...")
    user_embedding = sts_model.encode(translated_input, convert_to_tensor=True)
    print("Encoding evidence texts...")
    evidence_embeddings = sts_model.encode(evidence_texts, convert_to_tensor=True)

    # Calculate cosine similarities
    print("Calculating cosine similarities...")
    cos_scores = util.pytorch_cos_sim(user_embedding, evidence_embeddings)[0]

    # Find the evidence with the highest score
    print("Finding best matching evidence...")
    max_score_idx = torch.argmax(cos_scores).item()
    best_evidence = evidence_texts[max_score_idx]
    
    # Perform NLI
    result = perform_nli(best_evidence, translated_input)
    return result, best_evidence

# Example usage
if __name__ == "__main__":
    user_input = input("Enter a fact: ")
    result, evidence = infer_user_input(user_input, evidence_texts)
    print(f"Result: {result}")
    print(f"Evidence: {evidence}")