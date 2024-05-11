import torch
import optuna
import random
import transformers
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback, MarianMTModel, MarianTokenizer, BitsAndBytesConfig
from datasets import load_dataset, concatenate_datasets, interleave_datasets, DatasetDict, Dataset, ClassLabel
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
from optuna.trial import TrialState
from trl import SFTTrainer
from tqdm.auto import tqdm
import gc

# Liste des langues européennes supportées par HelsinkiNLP
european_langs = [
    'fr', 'es', 'it', 'de', 'nl', 'ro', 'el', 'bg', 'cs', 'da', 
    'et', 'fi', 'hu', 'mt', 'sk', 'sv', 
    'sq', 'hy', 'az', 'eu', 'ca', 'is', 'ga', 
    'mk', 'uk'
]

def get_tokenizer(model_name):
    if model_name == "llama":
        tokenizer = AutoTokenizer.from_pretrained("lightblue/suzume-llama-3-8B-multilingual", add_eos_token=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer
    elif model_name == "mt5":
        return AutoTokenizer.from_pretrained("google/mt5-xxl")
    else:
        raise ValueError("Unsupported model name")

def get_model(model_name, num_labels, quantization_config):
    if model_name == "llama":
        model = AutoModelForSequenceClassification.from_pretrained(
            "lightblue/suzume-llama-3-8B-multilingual", 
            num_labels=num_labels, 
            quantization_config=quantization_config,
            device_map={'':torch.cuda.current_device()}
        )
    elif model_name == "mt5":
        model = AutoModelForSequenceClassification.from_pretrained(
            "google/mt5-xxl", 
            num_labels=num_labels, 
            quantization_config=quantization_config,
            device_map={'':torch.cuda.current_device()}
        )
    else:
        raise ValueError("Unsupported model name")
    return model

class OptunaPruningCallback(TrainerCallback):
    """Hugging Face Trainer Callback for Optuna pruning."""

    def __init__(self, trial, monitor="eval_loss"):
        self._trial = trial
        self._monitor = monitor

    def on_evaluate(self, args, state, control, **kwargs):
        # Récupération de la métrique à surveiller
        logs = kwargs.get("logs", {})
        eval_loss = logs.get(self._monitor)

        # Informer Optuna que le trial doit être interrompu si la performance est insuffisante
        if eval_loss is None:
            return
        self._trial.report(eval_loss, step=state.global_step)
        if self._trial.should_prune():
            raise optuna.exceptions.TrialPruned()

def setup_translation_model(src_lang, target_lang):
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{target_lang}'
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        model.to("cuda")
        return tokenizer, model
    except Exception as e:
        print(f"Could not load model {model_name}: {e}")
        return None, None

def translate_batch(texts, tokenizer, model):
    try:
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.inference_mode():
            translated = model.generate(**inputs.to(model.device))
        return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    except Exception as e:
        print(f"Error during translation batch: {e}")
        return None


def convert_to_class_label(dataset):
    label_feature = ClassLabel(num_classes=3, names=['entailment', 'neutral', 'contradiction'])
    dataset = dataset.cast_column('labels', label_feature)
    return dataset

def translate_dataset(dataset, fields, target_langs, src_lang='en'):
    new_rows = []
    for target_lang in target_langs:
        tokenizer, model = setup_translation_model(src_lang, target_lang)
        progress_bar = tqdm(dataset, desc=f"Translating to {target_lang}")
        for example in progress_bar:
            try:
                translated_texts = translate_batch([example[field] for field in fields], tokenizer, model)
                new_row = {field: trans_text for field, trans_text in zip(fields, translated_texts)}
                new_row['labels'] = example['labels']
                new_row['language'] = target_lang
                new_rows.append(new_row)
            except Exception as e:
                print(f"Failed to translate to {target_lang}: {e}")
                progress_bar.set_description(f"Failed to translate to {target_lang}: {e}")

    # Reformats new_rows from a list of dictionaries to a dictionary of lists
    reformed_rows = {key: [dic[key] for dic in new_rows] for key in new_rows[0]}

    # Create a new dataset with the reformed rows
    translated_dataset = Dataset.from_dict(reformed_rows)
    dataset = convert_to_class_label(dataset)
    translated_dataset = convert_to_class_label(translated_dataset)
    return concatenate_datasets([dataset, translated_dataset])

def load_and_prepare_data():
    # Chargement et préparation des données XNLI pour le training en few-shot
    # dataset = load_dataset("ankitkupadhyay/XNLI", split="train")
    dataset = load_dataset("Gameselo/monolingual-wideNLI", split="train")
    dataset = dataset.shuffle(seed=1234).rename_column("label", "labels")
    # Sélectionner 8 échantillons par classe
    class_indices = {label: [] for label in set(dataset['labels'])}
    for index, label in enumerate(dataset['labels']):
        class_indices[label].append(index)
    sampled_indices = [index for label, indices in class_indices.items() for index in random.sample(indices, 8)]
    train_data = dataset.select(sampled_indices)
    
    # Chargement du dataset pour l'évaluation
    eval_dataset = load_dataset("Gameselo/monolingual-wideNLI", split="dev")
    eval_dataset = eval_dataset.shuffle(seed=42).select(range(8)).rename_column("label", "labels")  # Sélectionner 8 données aléatoires

    train_data = translate_dataset(train_data, ['premise', 'hypothesis'], european_langs)
    eval_dataset = translate_dataset(eval_dataset, ['premise', 'hypothesis'], european_langs)
    
    return train_data, eval_dataset

def tokenize_and_prepare_data(examples, model_name, tokenizer):
    if model_name == "mt5":
        # Utiliser un format de prompt spécifique pour mT5
        task_description = "Determine if the hypothesis is true based on the premise."
        texts = [f"Task: {task_description} Premise: {premise} Hypothesis: {hypothesis} </s>"
                 for premise, hypothesis in zip(examples['premise'], examples['hypothesis'])]
    else:
        # Format général utilisé pour d'autres modèles comme LLaMa
        texts = [f"Is this true? {premise} implies {hypothesis}"
                 for premise, hypothesis in zip(examples['premise'], examples['hypothesis'])]

    result = tokenizer(texts,
                       padding='max_length',
                       truncation=True,
                       max_length=1000,
                       return_overflowing_tokens=True)

    if 'labels' in examples:
        result['labels'] = examples['labels']

    sample_map = result.pop("overflow_to_sample_mapping")
    for key, values in examples.items():
        result[key] = [values[i] for i in sample_map]
    return result

def model_setup(trial, model_name):

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = get_model(model_name, num_labels=3, quantization_config=quantization_config)

    tokenizer = get_tokenizer(model_name)

    model.config.pad_token_id = model.config.eos_token_id

    # Additional PEFT configuration
    lora_config = LoraConfig(
        r=trial.suggest_categorical("lora_r", [4, 8, 12, 16]),
        lora_alpha=trial.suggest_categorical("lora_alpha", [16, 32, 64]),
        lora_dropout=trial.suggest_categorical("lora_dropout", [0.1, 0.15, 0.2]),
        bias="none"
    )

    return model, tokenizer, lora_config


def objective(trial, model_name):
    torch.cuda.empty_cache()
    gc.collect()
    
    train_data, eval_data = load_and_prepare_data()
    model, tokenizer, lora_config = model_setup(trial, model_name)

    train_data = train_data.map(lambda examples: tokenize_and_prepare_data(examples, model_name, tokenizer), batched=True)
    eval_data = eval_data.map(lambda examples: tokenize_and_prepare_data(examples, model_name, tokenizer), batched=True)
    
    gradient_accumulation_steps=trial.suggest_categorical("gradient_accumulation_steps", [4, 8])
    num_train_epochs=trial.suggest_categorical("epochs", [3, 4, 5])
    learning_rate=trial.suggest_categorical("learning_rate", [1e-4, 5e-5])
    lr_scheduler_type=trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "inverse_sqrt"])
    weight_decay=trial.suggest_categorical("weight_decay", [0.0, 0.01, 0.05])
    warmup_ratio=trial.suggest_float("warmup_ratio", 0.05, 0.3)

    # Ajout de plus d'hyperparamètres et configuration du pruning
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        per_device_train_batch_size=1,
	per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        optim="paged_adamw_8bit",
        lr_scheduler_type=lr_scheduler_type,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_dir='./logs',
        logging_steps=100,
        metric_for_best_model="eval_accuracy",
        save_strategy="no"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=lora_config,
        args=training_args,
        max_seq_length=1024,
        data_collator=transformers.DataCollatorWithPadding(tokenizer),
        formatting_func=tokenize_and_prepare_data,
        callbacks=[OptunaPruningCallback(trial)]
    )

    trainer.train()
    eval_result = trainer.evaluate()
    return eval_result["eval_accuracy"]

def model_setup_with_params(best_params, model_name):

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Configuration de LoRA (PEFT)
    LoraConfig(
        r = best_params["lora_r"],
        lora_alpha = best_params["lora_alpha"],
        lora_dropout = best_params["lora_dropout"],
        bias = "none"
    )
    
    model = get_model(model_name, num_labels=3, quantization_config=quantization_config)
    tokenizer = get_tokenizer(model_name)

    model.config.pad_token_id = model.config.eos_token_id

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    return model, tokenizer, lora_config

def train_with_best_params(best_params, model_name):
    torch.cuda.empty_cache()
    gc.collect()
    
    train_data, eval_data = load_and_prepare_data()

    model, tokenizer, lora_config = model_setup_with_params(best_params, model_name)

    train_data = train_data.map(lambda examples: tokenize_and_prepare_data(examples, model_name), batched=True)
    eval_data = eval_data.map(lambda examples: tokenize_and_prepare_data(examples, model_name), batched=True)

    training_args = TrainingArguments(
        output_dir="./final_results",
        evaluation_strategy="epoch",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=best_params["gradient_accumulation_steps"],
        optim="paged_adamw_8bit",
        lr_scheduler_type=best_params["lr_scheduler_type"],
        num_train_epochs=best_params["epochs"],
        learning_rate=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
        warmup_ratio=best_params["warmup_ratio"],
        logging_dir='./final_logs',
        logging_steps=100
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=lora_config,
        args=training_args,
        max_seq_length=1024,
        data_collator=transformers.DataCollatorWithPadding(tokenizer),
        formatting_func=tokenize_and_prepare_data,
        callbacks=[OptunaPruningCallback(trial)]
    )
    
    trainer.train()
    trainer.save_model("./best_model")  # Sauvegarder le modèle

def main():
    parser = argparse.ArgumentParser(description="Train a NLI fact-checker model")
    parser.add_argument("--model", type=str, choices=["llama", "mt5"], default="llama", help="Model to train (llama or mt5)")
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.model)

    model_name = args.model
    
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(lambda trial: objective(trial, model_name), n_trials=20, timeout=7200)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    train_with_best_params(study.best_trial.params)

if __name__ == "__main__":
    main()
