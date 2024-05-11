from datasets import load_dataset, concatenate_datasets, Dataset, Value, ClassLabel, Features, DatasetDict, interleave_datasets

def create_balanced_subset(dataset, num_samples_per_class, num_classes):
    # Assurez-vous que le dataset est filtré pour ne contenir que les classes nécessaires
    subsets = []
    for label_id in range(num_classes):
        # Filtrer par classe et prendre un échantillon aléatoire
        subset = dataset.filter(lambda example: example['label'] == label_id).shuffle(seed=42).select(range(min(num_samples_per_class, len(dataset.filter(lambda example: example['label'] == label_id)))))
        subsets.append(subset)
    # Interleave pour mélanger les classes
    balanced_subset = interleave_datasets(subsets)
    return balanced_subset

def process_generic_fever_like_data(dataset, label_mapping, default_label='NOT_ENOUGH_INFO', label_col='label'):
    premises = []
    hypotheses = []
    labels = []

    for item in dataset:
        premise = item.get('evidence', "No evidence provided.")
        hypothesis = item['claim']
        label_value = item.get(label_col, default_label)
        label = label_mapping.get(label_value, 1)  # Default to 'NOT_ENOUGH_INFO'

        premises.append(premise)
        hypotheses.append(hypothesis)
        labels.append(label)

    return Dataset.from_dict({
        'premise': premises,
        'hypothesis': hypotheses,
        'label': labels
    }, features=Features({
        'premise': Value('string'),
        'hypothesis': Value('string'),
        'label': ClassLabel(names=['entailment', 'neutral', 'contradiction'], id=None)
    }))

def rename_and_cast(dataset, old_new_names):
    dataset = dataset.rename_columns(old_new_names)
    if dataset == wnli:
        dataset = dataset.cast(new_features)
    return dataset

def map_labels(dataset, label_mapping):
    def map_function(example):
        example['label'] = label_mapping.get(example['label'], example['label'])
        return example
    return dataset.map(map_function)

def standardize_and_clean(dataset, columns_to_keep):
    columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
    return dataset.remove_columns(columns_to_remove)

# Initialisation des mappings
fever_label_mapping = {"SUPPORTS": 0, "REFUTES": 2, "NOT_ENOUGH_INFO": 1}
xfact_label_mapping = {"true": 0, "mostly_true": 0, "false": 2, "mostly_false": 2}
neutral_inv_mapping = {1: 0, 0: 1, 2: 2}
contradiction_rte_mapping = {1: 2, 0: 0, 2: 2}
scitail_label_mapping = {'neutral': 1, 'entails': 0, 'contradiction': 2}

# Définition des colonnes à conserver
columns_to_keep = ['premise', 'hypothesis', 'label']

# Chargement des datasets
snli = load_dataset('snli')
anli = load_dataset('facebook/anli')
mnli = load_dataset("nyu-mll/multi_nli")
fever = load_dataset("pietrolesci/nli_fever")
qnli = load_dataset("yangwang825/qnli")
wnli = load_dataset("gokuls/glue_augmented_wnli")
scitail = load_dataset("allenai/scitail", "tsv_format")
rte = load_dataset("yangwang825/rte")
climate_fever = load_dataset("Jasontth/climate_fever_plus")
vitaminc = load_dataset("tals/vitaminc")

# Traitement et standardisation de chaque dataset
datasets_list = [
    ("snli", snli), ("anli", anli), ("mnli", mnli), 
    ("fever", fever), ("qnli", qnli), ("wnli", wnli), 
    ("scitail", scitail), ("rte", rte), 
    ("climate_fever", climate_fever), ("vitaminc", vitaminc)
]

train_datasets = []
dev_datasets = []
test_datasets = DatasetDict()

num_classes = 3
new_features = snli["train"].features.copy()
new_features["label"] = ClassLabel(names=['entailment', 'neutral', 'contradiction'], id=None)


for dataset_name, dataset in datasets_list:
    if dataset in [climate_fever, vitaminc]:
        for key in dataset.keys():
            dataset[key] = process_generic_fever_like_data(dataset[key], fever_label_mapping)
    if dataset in [qnli, rte]:
        dataset = rename_and_cast(dataset, {"text1": "premise", "text2": "hypothesis"})
    if dataset in [wnli]:
        dataset = rename_and_cast(dataset, {"sentence1": "premise", "sentence2": "hypothesis"})
    dataset = map_labels(dataset, neutral_inv_mapping if dataset in [wnli] else contradiction_rte_mapping if dataset in [rte] else scitail_label_mapping if dataset in [scitail] else {})
    for key in dataset.keys():
        dataset[key] = standardize_and_clean(dataset[key], columns_to_keep)
    dataset = dataset.cast(new_features)
    
    if dataset_name == 'anli':
        for round_id in ["train_r1", "train_r2", "train_r3"]:
            split_data = dataset[round_id].train_test_split(test_size=0.05)
            print(f"Adding {len(split_data['test'])} items to dev_datasets from ANLI {round_id}")
            dev_datasets.append(split_data['test'])
            train_datasets.append(split_data['train'])
        for round_id in ["test_r1", "test_r2", "test_r3"]:
            test_datasets[round_id] = dataset[round_id]

    elif dataset_name == 'vitaminc':
        split_vitaminc_dev = dataset['validation'].train_test_split(test_size=0.1)
        print(f"Adding {len(split_vitaminc_dev['test'])} items to dev_datasets from VitaminC dev")
        dev_datasets.append(split_vitaminc_dev['test'])
        train_datasets.append(split_vitaminc_dev['train'])
        split_vitaminc_test = dataset['test'].train_test_split(test_size=0.1)
        test_datasets['vitaminc_test'] = split_vitaminc_test['test']
        train_datasets.append(split_vitaminc_test['train'])
    
    else:
        for key in dataset.keys():
            train_datasets.append(dataset[key])

if dev_datasets:
    final_dev_dataset = concatenate_datasets(dev_datasets)
    final_train_dataset = concatenate_datasets(train_datasets)
    final_datasets = DatasetDict({
        'train': final_train_dataset,
        'dev': final_dev_dataset,
        'test_anli_r1': test_datasets["test_r1"],
        'test_anli_r2': test_datasets["test_r2"],
        'test_anli_r3': test_datasets["test_r3"],
        'test_vitaminc': test_datasets["vitaminc_test"]
    })
    final_datasets.push_to_hub("Gameselo/monolingual-wideNLI", token='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')
else:
    print("No datasets added to dev_datasets, check data availability and split conditions.")