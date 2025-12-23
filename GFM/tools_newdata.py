from transformers import Trainer
import torch
from tqdm import tqdm
import os
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def load_preprocessed_data(file_path):
    """Load previously saved preprocessing results."""
    if os.path.exists(file_path):
        preprocessed_data = torch.load(file_path)
        print(f"Preprocessed dataset loaded from {file_path}")
        return preprocessed_data
    else:
        raise FileNotFoundError(f"{file_path} not found!")

def preprocess_and_save(dataset, cluster_dict, output_dir):
    """Convert the dataset into padded tensors, then persist them for reuse."""
    os.makedirs(output_dir, exist_ok=True)
    preprocessed_data = []
    MAX_SEQ_LENGTH = 8192
    for item in tqdm(dataset, desc="Preprocessing dataset"):
        seq_ids = torch.LongTensor(sum(item['feature_index'], []))
        cluster_ids = torch.LongTensor(sum(item['cluster_id'], [])) + 5 

        original_length = seq_ids.size(0)
        target_length = ((original_length + 62) // 63) * 63
        pad_length = target_length - original_length

        seq_ids_padded = torch.cat([seq_ids, torch.full((pad_length,), -100, dtype=seq_ids.dtype)])
        cluster_ids_padded = torch.cat([cluster_ids, torch.full((pad_length,), 0, dtype=cluster_ids.dtype)])
        
        seq_ids_result = []
        cluster_ids_result = []

        step = 63
        for i in range(0,len(seq_ids_padded), step):
            seq_ids_result.append(-100)
            cluster_ids_result.append(1)

            seq_ids_result.extend(seq_ids_padded[i:i+step].tolist())
            cluster_ids_result.extend(cluster_ids_padded[i:i+step].tolist())

        seq_ids_result = torch.LongTensor(seq_ids_result)
        cluster_ids_result = torch.LongTensor(cluster_ids_result)
        
        if seq_ids_result.size(0) > MAX_SEQ_LENGTH:
            seq_ids_result = seq_ids_result[:MAX_SEQ_LENGTH]
            cluster_ids_result = cluster_ids_result[:MAX_SEQ_LENGTH]
            
        preprocessed_data.append({
            'seq_ids': seq_ids_result,
            'cluster_ids': cluster_ids_result,
        })

    save_path = os.path.join(output_dir, "preprocessed_data.pt")
    torch.save(preprocessed_data, save_path)
    print(f"Preprocessed dataset saved to {save_path}")

    return preprocessed_data


class DataCollatorLM:
    def __init__(self, config, embedding, mlm = True, mlm_probability = 0.15):
        self.config = config
        self.embedding = embedding
        self.embed_shape = embedding.shape
        self.mlm = mlm
        self.mlm_probability = mlm_probability
    def __call__(self, batch):

        seq_ids = [item['seq_ids'] for item in batch]
        cluster_ids = [item['cluster_ids'] for item in batch]

        seq_ids = pad_sequence(seq_ids, batch_first=True, padding_value=self.config.pad_token_id)
        cluster_ids = pad_sequence(cluster_ids, batch_first=True, padding_value=self.config.pad_token_id)


        masked_seq_ids = seq_ids.clone()
        inputs_embeds, labels = self.mask_tokens(masked_seq_ids, cluster_ids)
        attention_mask = (cluster_ids != self.config.pad_token_id).long()
        return {'inputs_embeds': inputs_embeds, 'attention_mask': attention_mask, 'labels': labels, 'cluster_ids': cluster_ids}

    def mask_tokens(self, inputs, clusters):
        labels = clusters.clone()
        
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = labels < self.config.num_special_tokens

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)            
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        labels[~masked_indices] = -100  # We only compute loss on masked tokens, -100 is default for CE compute

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.config.pad_token_id

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(0, self.embed_shape[0], labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        
        inputs_adjusted = inputs
        
        valid_indices = inputs_adjusted >= 0
        
        inputs_clipped = torch.clamp(inputs_adjusted, min=0)
        
        inputs_embeds = self.embedding[inputs_clipped]
        
        inputs_embeds[~valid_indices] = 0

        return torch.tensor(inputs_embeds), labels

class CustomTrainer(Trainer):
    def _remove_unused_columns(self, dataset, description=None):
        return dataset
    def get_train_dataloader(self):
        train_dataset = self.train_dataset

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=8
        )

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=8
        )
