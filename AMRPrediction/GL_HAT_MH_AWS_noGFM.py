import gc
import os
import warnings

import numpy as np
import torch
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

from Model_MH_AWS_noGFM import HATConfig, HATForMaskedLM, AMRPredictor
from tools_newdata_amr import DataCollator, compute_metrics, preprocess_and_save, load_preprocessed_data

warnings.simplefilter(action='ignore', category=FutureWarning)

np.int = int
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:192"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def print_cuda_memory():
    print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e6} MB", flush = True)
    print(f"Allocated memory: {torch.cuda.memory_allocated(0) / 1e6} MB", flush = True)
    print(f"Cached memory: {torch.cuda.memory_reserved(0) / 1e6} MB\n", flush = True)

gc.collect()
torch.cuda.empty_cache()
print_cuda_memory()

embedding = np.memmap(f"./embed_protein.dat", dtype='float32', mode='r+', shape=(11179545, 1280))
target_row = 146198
valid_rows = np.where(~np.isnan(embedding).any(axis=1))[0]
random_row = np.random.choice(valid_rows)
embedding[target_row] = embedding[random_row]
print(f"Replaced row {target_row} with values from row {random_row}.")

class CustomTrainer(Trainer):
    def _remove_unused_columns(self, dataset, description=None):
        # Keep all dataset columns because the model expects custom keys.
        return dataset

t = 1200
g = 32
for r in [1e-5]:
    for w in [0.1]:
        output_dir_train = f'dataset/train_preprocessing_amr_fl/'
        output_dir_valid = f'dataset/valid_preprocessing_amr_fl/'
        if os.path.exists(output_dir_train):
            train_preprocessed_data = load_preprocessed_data(output_dir_train + 'preprocessed_data.pt')
            val_preprocessed_data = load_preprocessed_data(output_dir_valid + 'preprocessed_data.pt')
        else:
            train_dataset = load_from_disk(f'dataset_ast_protein_filtered_label/train')
            val_dataset = load_from_disk(f'dataset_ast_protein_filtered_label/validation')
            train_preprocessed_data = preprocess_and_save(train_dataset, output_dir_train)
            val_preprocessed_data = preprocess_and_save(val_dataset,  output_dir_valid)
    

        training_args = TrainingArguments(
            output_dir=f"./results_nd2_ep235_gf_noamr_ga{g}_it{t}_lr{r}_wd{w}_gene_lnmhattli_temp05_wopt",
            save_steps=4590,
            save_strategy='steps',
            save_total_limit=1,     # Keep only the best checkpoint.
            num_train_epochs=t,
            eval_steps=4590,
            evaluation_strategy='steps',
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            learning_rate=r,
            warmup_steps=10000,
            weight_decay=w,
            logging_steps=4590,
            logging_strategy='steps',
            report_to=None,
            load_best_model_at_end = True,
            metric_for_best_model="eval_loss",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            gradient_accumulation_steps = g
        )
        
    
        genome_config = HATConfig.from_pretrained("config_hat_mlm.txt")
        genome_model = HATForMaskedLM(config = genome_config).to('cuda')
        # genome_model.load_state_dict(torch.load('./GFM/results_genome_newdata-epoch235/checkpoint-7650190/pytorch_model.bin'))
        model = AMRPredictor(genome_model.hi_transformer).to('cuda')
        del genome_model
        gc.collect()
        torch.cuda.empty_cache()
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_preprocessed_data,
            eval_dataset=val_preprocessed_data,
            data_collator=DataCollator(embedding = embedding),
            compute_metrics=compute_metrics,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=50)]
        )
            
        try:
            trainer.train()
        finally:
            del model
            del trainer
            del train_preprocessed_data
            del val_preprocessed_data
            gc.collect()
            torch.cuda.empty_cache()
            print_cuda_memory()
