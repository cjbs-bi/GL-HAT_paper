import os
from tqdm import tqdm
import numpy as np
from transformers import TrainingArguments
from datasets import load_from_disk
from Model_GFM import HATConfig, HATForMaskedLM
from tools_newdata import DataCollatorLM, CustomTrainer, preprocess_and_save, load_preprocessed_data
import pickle
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:192"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# embed_protein.dat: Embedding vectors generated with ESM-2
embedding = np.memmap(f"./embed_protein.dat", dtype='float32', mode='r+', shape=(11179545, 1280))

# Replace any NaN-containing embedding row by copying a randomly selected valid row
target_row = 146198
valid_rows = np.where(~np.isnan(embedding).any(axis=1))[0]
random_row = np.random.choice(valid_rows)
embedding[target_row] = embedding[random_row]
print(f"Replaced row {target_row} with values from row {random_row}.")

# cluster_dict maps each protein sequence ID (key) to its K-Means cluster label (value)
file_name = 'cluster_dict.pkl'
if os.path.exists(file_name):
    with open(file_name,'rb') as f:
        cluster_dict = pickle.load(f)
else:
    cluster_dict = {}
    with open("kmeans.txt") as kmeans_file:
        for line in tqdm(kmeans_file, total = 11179545):
            info = line.rstrip().split('\t')
            cluster_dict[int(info[0])] = int(info[1])

        # saving the newly built cluster_dict
        with open(file_name, 'wb') as f:
            pickle.dump(cluster_dict, f)
        print(f"cluster_dict is saved: {file_name}")
# Preprocess the train/validation/test datasets
output_dir_train = 'dataset/train_preprocessing'
output_dir_valid = 'dataset/valid_preprocessing'
output_dir_test = 'dataset/test_preprocessing' 
if os.path.exists(output_dir_train):
    # Load preprocessed datasets
    train_preprocessed_data = load_preprocessed_data(output_dir_train + '/preprocessed_data.pt')
    val_preprocessed_data = load_preprocessed_data(output_dir_valid + '/preprocessed_data.pt')
    test_preprocessed_data = load_preprocessed_data(output_dir_test + '/preprocessed_data.pt')
else:
    # Load raw datasets
    train_dataset = load_from_disk('dataset_ast_protein/train') 
    val_dataset = load_from_disk('dataset_ast_protein/validation')
    test_dataset = load_from_disk('dataset_ast_protein/test')
    # Dataset preprocessing
    train_preprocessed_data = preprocess_and_save(train_dataset, cluster_dict, output_dir_train)
    val_preprocessed_data = preprocess_and_save(val_dataset, cluster_dict, output_dir_valid)
    test_preprocessed_data = preprocess_and_save(test_dataset, cluster_dict, output_dir_test)

training_args = TrainingArguments(
    output_dir=f"./results_genome_newdata",
    num_train_epochs=600,
    evaluation_strategy='epoch',
    logging_strategy="epoch",     
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=1e-4,
    warmup_steps=1000,
    weight_decay=0.01,
    remove_unused_columns=False,
    save_safetensors=False,
    save_strategy='epoch',
    save_total_limit=1,                     
    load_best_model_at_end=True,           
    metric_for_best_model="eval_loss",   
    greater_is_better=False,            

    
)
config = HATConfig.from_pretrained('config_hat_mlm.txt')
model = HATForMaskedLM(config = config)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_preprocessed_data,
    eval_dataset=val_preprocessed_data,
    data_collator=DataCollatorLM(config = config, embedding = embedding, mlm_probability = 0.15)
)

trainer.train()
