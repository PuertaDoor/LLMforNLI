import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import T5ForConditionalGeneration, T5Tokenizer, MarianMTModel, MarianTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Load datasets
datasets = {
    "FEVER": ("pietrolesci/nli_fever", ["train", "dev", "test"]),
    "CLIMATE-FEVER": ("Jasontth/climate_fever_plus", ["climate_fever"]),
    "XFACT": ("metaeval/x-fact", ["train", "dev", "test"]),
    "SciTail": ("allenai/scitail", ["train", "validation", "test"], "tsv_format"),
    "VitaminC": ("tals/vitaminc", ["train", "validation", "test"])
}

def load_datasets(datasets):
    data = {}
    print("Loading datasets...")
    for name, details in tqdm(datasets.items()):
        path, splits = details[0], details[1]
        format_type = details[2] if len(details) > 2 else None
        dataset = [load_dataset(path, format_type, split=split) for split in splits] if format_type else [load_dataset(path, split=split) for split in splits]
        data[name] = pd.concat([d.to_pandas() for d in dataset])
    print("Datasets loaded.")
    return data

# Function to check labels and map them if necessary
def check_and_map_labels(data):
    label_map = {
        "SUPPORTS": 0,
        "entailment": 0,
        "entails": 0,
        "REFUTES": 2,
        "contradiction": 2,
        "NOT ENOUGH INFO": 1,
        "NOT_ENOUGH_INFO": 1,
        "neutral": 1,
        "false": 2,
        "true": 0,
        "partly true/misleading": 1,
        "mostly true": 0,
        "mostly false": 2,
        "half true": 1,
        "complicated": 1,
        "complicated/hard to categorise": 1,
        "other": 1,
        "not available": 1
    }

    valid_labels = {0, 1, 2}

    column_map = {
        "FEVER": "fever_gold_label",
        "CLIMATE-FEVER": "evidence_label",
        "XFACT": "label",
        "SciTail": "label",
        "VitaminC": "label"
    }

    for dataset_name, df in data.items():
        label_column = column_map[dataset_name]
        df['label'] = df[label_column].map(label_map)
        df['label'] = df['label'].fillna(-1).astype(int)
        if not set(df['label'].unique()).issubset(valid_labels):
            raise ValueError(f"Dataset {dataset_name} contains invalid labels after mapping.")
    
    return data

data = load_datasets(datasets)
data = check_and_map_labels(data)

# Map column names to the respective dataset for extracting evidence/hypothesis texts
evidence_column_map = {
    "FEVER": "hypothesis",
    "CLIMATE-FEVER": "evidence",
    "XFACT": "evidence",
    "SciTail": "hypothesis",
    "VitaminC": "evidence"
}

# Collect all evidence texts into one list
print("Preparing evidence texts...")
evidence_texts = []
for dataset_name, df in data.items():
    column = evidence_column_map[dataset_name]
    evidence_texts.extend(df[column].dropna().apply(str).tolist())

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
flan_t5_model = T5ForConditionalGeneration.from_pretrained(flan_t5_model_name, device_map="auto")
print("NLI model loaded.")

# Prepare the translation model for any language to English
translation_model_name = "Helsinki-NLP/opus-mt-mul-en"
print("Loading translation model...")
translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
translation_model = MarianMTModel.from_pretrained(translation_model_name)
print("Translation model loaded.")

# Claims verificados, no hace falta calcular de nuevo
# Calculer seulement une fois et conserver les embeddings (à voir comment)

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
    pool = sts_model.start_multi_process_pool()
    evidence_embeddings = sts_model.encode_multi_process(evidence_texts, pool, batch_size=1000) # numpy array
    sts_model.stop_multi_process_pool(pool)

    evidence_embeddings = torch.from_numpy(evidence_embeddings).to(user_embedding.device) # conversion to tensor

    # Calculate cosine similarities
    print("Calculating cosine similarities...")
    cos_scores = util.pytorch_cos_sim(user_embedding, evidence_embeddings)[0]

    # Find the evidence with the highest score
    print("Finding best matching evidence...")
    max_score_idx = torch.argmax(cos_scores).item()
    best_evidence = evidence_texts[max_score_idx]
    
    # Translate the best evidence to English
    translated_evidence = translate_to_english(best_evidence)

    # Perform NLI
    result = perform_nli(translated_input, translated_evidence)
    return result, best_evidence

# Example usage
if __name__ == "__main__":
    user_input = input("Enter a fact: ")
    result, evidence = infer_user_input(user_input, evidence_texts)
    print(f"Result: {result}")
    print(f"Evidence: {evidence}")