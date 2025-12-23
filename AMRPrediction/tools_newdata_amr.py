from transformers import AutoModel, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, BigBirdModel, PretrainedConfig, TrainerCallback
import time
import torch
from tqdm import tqdm
import os
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import json
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve, auc, confusion_matrix, average_precision_score

class LossLoggerCallback(TrainerCallback):
    def __init__(self, output_file="losses.json"):
        self.train_losses = []
        self.eval_losses = []
        self.output_file = output_file  # 저장할 파일 이름

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs and "eval_loss" not in logs:  # Train Loss 저장
                self.train_losses.append(logs["loss"])
            if "eval_loss" in logs:  # Eval Loss 저장
                self.eval_losses.append(logs["eval_loss"])

    def on_epoch_end(self, args, state, control, **kwargs):
        # Train이 끝난 후 loss를 파일로 저장
        with open(self.output_file, "w") as f:
            json.dump({
                "train_losses": self.train_losses,
                "eval_losses": self.eval_losses
            }, f, indent=4)
        print(f"Losses saved to {self.output_file}")
                
def load_preprocessed_data(file_path):
    """
    전처리된 데이터를 불러오기
    """
    if os.path.exists(file_path):
        preprocessed_data = torch.load(file_path)
        print(f"Preprocessed dataset loaded from {file_path}")
        return preprocessed_data
    else:
        raise FileNotFoundError(f"{file_path} not found!")

def preprocess_and_save(dataset, output_dir):
    """
    데이터셋을 전처리하여 리스트로 반환loss_logger
    """
    os.makedirs(output_dir, exist_ok=True)  # 저장 디렉토리 생성
    preprocessed_data = []
    # seq_ids_test = []
    # cluster_ids_check = []
    MAX_SEQ_LENGTH = 8192
    for item in tqdm(dataset, desc="Preprocessing dataset"):
        seq_ids = torch.LongTensor(sum(item['feature_index'], []))
        cluster_ids = torch.LongTensor(sum(item['cluster_id'], [])) + 5 
        labels = torch.Tensor(item['ast']).float()

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

        # 리스트를 하나의 LongTensor로 변환
        seq_ids_result = torch.LongTensor(seq_ids_result)
        cluster_ids_result = torch.LongTensor(cluster_ids_result)
        # seq_ids_test.append(seq_ids_result)

        
        if seq_ids_result.size(0) > MAX_SEQ_LENGTH:
            seq_ids_result = seq_ids_result[:MAX_SEQ_LENGTH]
            cluster_ids_result = cluster_ids_result[:MAX_SEQ_LENGTH]
            
        preprocessed_data.append({
            'seq_ids': seq_ids_result,
            'cluster_ids': cluster_ids_result,
            'labels' : labels
            # 'contig_index': contig_index,
            # 'feature_index': feature_index,
        })
        # 모든 cluster_ids를 하나의 텐서로 합치기
    
    # all_seq_ids = torch.cat(seq_ids_test)

    # # # 유니크한 값들 얻기
    # unique_values = torch.unique(all_seq_ids)
    # 전처리된 데이터를 저장
    save_path = os.path.join(output_dir, "preprocessed_data.pt")
    torch.save(preprocessed_data, save_path)
    print(f"Preprocessed dataset saved to {save_path}")

    return preprocessed_data

def preprocess_and_save_validate(dataset, cluster_dict, output_dir):
    """
    데이터셋을 전처리하여 리스트로 반환loss_logger
    """
    os.makedirs(output_dir, exist_ok=True)  # 저장 디렉토리 생성
    preprocessed_data = []
    # seq_ids_test = []
    # cluster_ids_check = []
    MAX_SEQ_LENGTH = 8192
    for item in tqdm(dataset, desc="Preprocessing dataset"):
        name = item['name']
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

        # 리스트를 하나의 LongTensor로 변환
        seq_ids_result = torch.LongTensor(seq_ids_result)
        cluster_ids_result = torch.LongTensor(cluster_ids_result)
        # seq_ids_test.append(seq_ids_result)

        
        if seq_ids_result.size(0) > MAX_SEQ_LENGTH:
            seq_ids_result = seq_ids_result[:MAX_SEQ_LENGTH]
            cluster_ids_result = cluster_ids_result[:MAX_SEQ_LENGTH]
            
        preprocessed_data.append({
            'name' : name,
            'seq_ids': seq_ids_result,
            'cluster_ids': cluster_ids_result,
            # 'contig_index': contig_index,
            # 'feature_index': feature_index,
        })
        # 모든 cluster_ids를 하나의 텐서로 합치기
    
    # all_seq_ids = torch.cat(seq_ids_test)

    # # # 유니크한 값들 얻기
    # unique_values = torch.unique(all_seq_ids)
    # 전처리된 데이터를 저장
    save_path = os.path.join(output_dir, "preprocessed_data.pt")
    torch.save(preprocessed_data, save_path)
    print(f"Preprocessed dataset saved to {save_path}")

    return preprocessed_data

class DataCollatorLM_validate:
    def __init__(self, config, embedding, mlm = True, mlm_probability = 0.15):
        self.config = config
        self.embedding = embedding
        self.embed_shape = embedding.shape
        self.mlm = mlm
        self.mlm_probability = mlm_probability
    def __call__(self, batch):
        name = [item['name'] for item in batch]
        seq_ids = [item['seq_ids'] for item in batch]
        cluster_ids = [item['cluster_ids'] for item in batch]

        seq_ids = pad_sequence(seq_ids, batch_first=True, padding_value=self.config.pad_token_id)
        cluster_ids = pad_sequence(cluster_ids, batch_first=True, padding_value=self.config.pad_token_id)


        # start_time = time.time()  # 시작 시간 기록 
        masked_seq_ids = seq_ids.clone()
        inputs_embeds, labels = self.mask_tokens(masked_seq_ids, cluster_ids)
        attention_mask = (cluster_ids != self.config.pad_token_id).long()
        # print(f"part3: {time.time() - start_time:.4f} seconds")
        # Print batch size and device for debugging
        return {'name' : name, 'seq_ids': seq_ids, 'inputs_embeds': inputs_embeds, 'attention_mask': attention_mask, 'labels': labels, 'cluster_ids': cluster_ids, 'masked_seq_ids': masked_seq_ids}#, 'masked_seq_ids': masked_seq_ids, 'seq_ids': seq_ids}

    def mask_tokens(self, inputs, clusters):
        labels = clusters.clone()
        
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        # special_tokens_mask = [[1 if v < self.config.num_special_tokens else 0 for v in val] for val in labels.tolist()]
        special_tokens_mask = labels < self.config.num_special_tokens

        # probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)            
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        labels[~masked_indices] = -100  # We only compute loss on masked tokens, -100 is default for CE compute

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.config.pad_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(0, self.embed_shape[0], labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        
        # inputs_adjusted = inputs - 1
        inputs_adjusted = inputs
        
        # Determine valid indices (indices >= 0)
        valid_indices = inputs_adjusted >= 0
        
        # Clamp valid indices to the range [0, self.embed_shape[0] - 1]
        inputs_clipped = torch.clamp(inputs_adjusted, min=0)
        
        # Retrieve embeddings using valid indices
        inputs_embeds = self.embedding[inputs_clipped]
        
        # Set embeddings corresponding to invalid indices to 0
        inputs_embeds[~valid_indices] = 0

        return torch.tensor(inputs_embeds), labels

class DataCollator:
    def __init__(self, embedding, pad_token_id = 0):
        self.pad_token_id = pad_token_id
        self.embedding = embedding
        self.embed_shape = embedding.shape
        if isinstance(self.embedding, (np.ndarray, np.memmap)):
            self.embedding = torch.tensor(self.embedding, dtype=torch.float)
        self.pad_keys = ["seq_ids", "cluster_ids"]
        self.padding_values = {"seq_ids": 0, "cluster_ids": 0}

    def __call__(self, batch):    
        batch_tensors = {key: [item[key] for item in batch] for key in batch[0]}

        padded_tensors = {
        key: pad_sequence(batch_tensors[key], batch_first=True, padding_value=self.padding_values[key])
        for key in self.pad_keys
    }
        seq_ids_tensor = padded_tensors["seq_ids"] - 1  # (row - 1)

        valid_idx = seq_ids_tensor.clamp(min=0)  # (i >= 0 처리)
        inputs_embeds = torch.where(seq_ids_tensor.unsqueeze(-1) >= 0,  
                                    self.embedding[valid_idx],  
                                    torch.zeros_like(self.embedding[0]))  # 
        attention_mask = (padded_tensors["cluster_ids"] != 0).long()
        padded_tensors.update({
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "labels" : batch[0]['labels'].unsqueeze(0)
    })
        return padded_tensors

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


        # start_time = time.time()  # 시작 시간 기록 
        masked_seq_ids = seq_ids.clone()
        inputs_embeds, labels = self.mask_tokens(masked_seq_ids, cluster_ids)
        attention_mask = (cluster_ids != self.config.pad_token_id).long()
        # print(f"part3: {time.time() - start_time:.4f} seconds")
        # Print batch size and device for debugging
        return {'inputs_embeds': inputs_embeds, 'attention_mask': attention_mask, 'labels': labels, 'cluster_ids': cluster_ids}#, 'masked_seq_ids': masked_seq_ids, 'seq_ids': seq_ids}

    def mask_tokens(self, inputs, clusters):
        labels = clusters.clone()
        
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        # special_tokens_mask = [[1 if v < self.config.num_special_tokens else 0 for v in val] for val in labels.tolist()]
        special_tokens_mask = labels < self.config.num_special_tokens

        # probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)            
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        labels[~masked_indices] = -100  # We only compute loss on masked tokens, -100 is default for CE compute

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.config.pad_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(0, self.embed_shape[0], labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        
        # inputs_adjusted = inputs - 1
        inputs_adjusted = inputs
        
        # Determine valid indices (indices >= 0)
        valid_indices = inputs_adjusted >= 0
        
        # Clamp valid indices to the range [0, self.embed_shape[0] - 1]
        inputs_clipped = torch.clamp(inputs_adjusted, min=0)
        
        # Retrieve embeddings using valid indices
        inputs_embeds = self.embedding[inputs_clipped]
        
        # Set embeddings corresponding to invalid indices to 0
        inputs_embeds[~valid_indices] = 0

        return torch.tensor(inputs_embeds), labels

class CustomTrainer(Trainer):
    def _remove_unused_columns(self, dataset, description=None):
        # Override this method to stop it from removing columns that are not in the model's forward method
        return dataset
    def get_train_dataloader(self):
        """
        Returns the training DataLoader with `num_workers` properly configured.
        """
        train_dataset = self.train_dataset

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=8  # 적절한 CPU 코어 수로 조정
        )

    def get_eval_dataloader(self, eval_dataset=None):
        """
        Returns the evaluation DataLoader with `num_workers` properly configured.
        """
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=8  # 적절한 CPU 코어 수로 조정
        )

def compute_metrics(eval_pred):
    # Unpack predictions and labels
    logits, labels = eval_pred

    # Convert logits to probabilities
    probabilities = torch.sigmoid(torch.tensor(logits)).numpy()
    # labels.unsqueeze(0)
    num_labels = labels.shape[1]
    
    # Initialize lists for storing scores per label
    prauc_scores = []
    auroc_scores = []
    f1_scores = []

    # Calculate metrics for each label
    for i in range(num_labels):
        prob, label = probabilities[:, i], labels[:, i]
        mask = label >= 0

        prob = prob[mask]
        label = label[mask]
        
        if np.count_nonzero(label == 1) > 0:
            # PRAUC (Average Precision)
            prauc = average_precision_score(label, prob)
            prauc_scores.append(prauc)
            
        else:
            # If calculation is impossible (e.g., no positive samples), append NaN
            prauc_scores.append(float('nan'))

        try:
            # AUROC
            auroc = roc_auc_score(label, prob)
            auroc_scores.append(auroc)
            
            # F1 Score (binary classification per label)
            f1 = f1_score(label, prob > 0.5)
            f1_scores.append(f1)
            
        except ValueError:
            auroc_scores.append(float('nan'))
            f1_scores.append(float('nan'))
    
    # Calculate macro averages, ignoring NaN values
    macro_prauc = np.nanmean(prauc_scores)
    macro_auroc = np.nanmean(auroc_scores)
    macro_f1 = np.nanmean(f1_scores)

    return {
        'eval_macro_prauc': macro_prauc,
        'eval_macro_auroc': macro_auroc,
        'eval_macro_f1': macro_f1
    }

def count_model_parameters(model):
    total_params = 0
    trainable_params = 0

    for param in model.parameters():
        # Multiply the size of each dimension to get the total number of elements in the parameter tensor
        num_params = param.numel()
        total_params += num_params

        if param.requires_grad:
            # If the parameter requires gradient, it's trainable
            trainable_params += num_params

    return total_params, trainable_params    
