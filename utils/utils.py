import torch
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple
import pandas as pd
from utils.datasetv2 import encode_text_to_tokens
from torch.utils.data import DataLoader
class GaussianNoise(object):
    """ Add Gaussian noise to a tensor.
    
    Args:
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.
    
    Returns:
        Tensor with added Gaussian noise.
    """
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.from_numpy(np.array(tensor))
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise

def create_vocabulary(texts: List[str], min_freq: int = 1):
    """Create vocabulary from text data
    
    Args:
        texts (List[str]): List of text strings
        min_freq (int): Minimum frequency for a word to be included in the vocabulary
        
        Returns:
        vocab_map (Dict[str, int]): Mapping from words to indices"""
    word_counts = Counter()
    for text in texts:
        words = text.strip().split()
        word_counts.update(words)
    
    vocab_words = [word for word, count in word_counts.items() if count >= min_freq]
    vocab_words = sorted(vocab_words)
    
    special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']
    vocab_list = special_tokens + vocab_words
    
    vocab_map = {word: idx for idx, word in enumerate(vocab_list)}
    inv_vocab_map = {idx: word for idx, word in enumerate(vocab_list)}
    
    print(f"Created vocabulary with {len(vocab_list)} tokens")
    return vocab_map, inv_vocab_map, vocab_list

def create_processed_dataframe(df: pd.DataFrame, target_column: str, vocab_map: Dict[str, int]):
    """Create processed dataframe with encoding
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Column name containing text data
        vocab_map (Dict[str, int]): Vocabulary mapping from words to indices
        
        Returns:
        pd.DataFrame: Processed dataframe with encoded text"""
    
    processed_df = df.copy()
    
    encoded_texts = []
    for _, row in processed_df.iterrows():
        text = row[target_column]
        encoded = encode_text_to_tokens(text, vocab_map)
        encoded_texts.append(encoded)
    
    processed_df['enc'] = encoded_texts
    return processed_df

def load_and_process_text_data(train_csv: str, dev_csv: str, target_column: str = 'gloss', additional_pose_files: List[str] = None):
    """Load and process text data
    
    Args:
        train_csv (str): Path to training CSV file
        dev_csv (str): Path to development CSV file
        target_column (str): Column name containing text data
        
        Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int], Dict[int, str], List[str]]: Processed dataframes and vocabulary info"""
    train_df = pd.read_csv(train_csv, delimiter="|")
    dev_df = pd.read_csv(dev_csv, delimiter="|")
    
    train_df = train_df.dropna(subset=['id', target_column])
    dev_df = dev_df.dropna(subset=['id', target_column])
    
    print(f"Loaded {len(train_df)} train, {len(dev_df)} dev samples")
    
    all_text = train_df[target_column].tolist()
    
    if additional_pose_files:
        import pickle
        import os
        for pkl_path in additional_pose_files:
            if os.path.exists(pkl_path):
                try:
                    with open(pkl_path, 'rb') as f:
                        data = pickle.load(f)
                        for key, val in data.items():
                            if isinstance(val, dict) and 'label' in val:
                                all_text.append(val['label'])
                except Exception as e:
                    print(f"Error loading {pkl_path} for vocabulary: {e}")

    vocab_map, inv_vocab_map, vocab_list = create_vocabulary(all_text)
    
    train_processed = create_processed_dataframe(train_df, target_column, vocab_map)
    dev_processed = create_processed_dataframe(dev_df, target_column, vocab_map)
    
    return train_processed, dev_processed, vocab_map, inv_vocab_map, vocab_list

def send_inputs_to_device(batch, device):
    """Send inputs to device
    
    Args:
        batch (dict): Batch of data containing tensors
        device (str or torch.device): Device to send tensors to
        
        Returns:
        dict: Batch with tensors moved to the specified device"""
    result = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device=device)
        else:
            result[key] = value
    return result


def decode_predictions(logits, vocab_info, max_length=50):
    """
    Decode model logits to text predictions

    Args:
        logits (torch.Tensor): Model output logits of shape [batch_size, seq_len, vocab_size]
        vocab_info (Dict): Vocabulary information containing 'inv_vocab_map'
        max_length (int): Maximum length of the predicted sequence

    Returns:
        List[str]: List of predicted text sequences

    """
    # Get the predicted token IDs (greedy decoding)
    predicted_ids = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]
    
    predictions = []
    inv_vocab_map = vocab_info['inv_vocab_map']
    
    for batch_idx in range(predicted_ids.size(0)):
        pred_sequence = predicted_ids[batch_idx].cpu().numpy()
        
        # Convert token IDs to words
        words = []
        for token_id in pred_sequence:
            if token_id in inv_vocab_map:
                word = inv_vocab_map[token_id]
                
                # Stop at EOS token or skip special tokens
                if word == '<eos>':
                    break
                elif word not in ['<pad>', '<bos>', '<unk>']:
                    words.append(word)
        
        predictions.append(' '.join(words))
    
    return predictions

def decode_labels(labels, vocab_info):
    """
    Decode label tensor to ground truth text

    Args:
        labels (torch.Tensor): Ground truth labels of shape [batch_size, seq_len]
        vocab_info (Dict): Vocabulary information containing 'inv_vocab_map'
    Returns:
        List[str]: List of ground truth text sequences

    """
    ground_truths = []
    inv_vocab_map = vocab_info['inv_vocab_map']
    
    for batch_idx in range(labels.size(0)):
        label_sequence = labels[batch_idx].cpu().numpy()
        
        words = []
        for token_id in label_sequence:
            if token_id in inv_vocab_map:
                word = inv_vocab_map[token_id]
                # Skip special tokens
                if word not in ['<pad>', '<bos>', '<eos>', '<unk>']:
                    words.append(word)
        
        ground_truths.append(' '.join(words))
    
    return ground_truths


def test_load_and_process_text_data(train_csv: str, dev_csv: str, test_csv: str, target_column: str = 'gloss', additional_pose_files: List[str] = None):
    """Load and process text data"""
    train_df = pd.read_csv(train_csv, delimiter="|")
    dev_df = pd.read_csv(dev_csv, delimiter="|")
    test_df = pd.read_csv(test_csv, delimiter="|")
    
    train_df = train_df.dropna(subset=['id', target_column])
    dev_df = dev_df.dropna(subset=['id', target_column])
    test_df = test_df.dropna(subset=['id'])
    
    print(f"Loaded {len(train_df)} train, {len(dev_df)} dev, {len(test_df)} test samples")
    
    # Use training data to create vocabulary (consistent with training)
    all_text = train_df[target_column].tolist()
    
    if additional_pose_files:
        import pickle
        import os
        for pkl_path in additional_pose_files:
            if os.path.exists(pkl_path):
                try:
                    with open(pkl_path, 'rb') as f:
                        data = pickle.load(f)
                        for key, val in data.items():
                            if isinstance(val, dict) and 'label' in val:
                                all_text.append(val['label'])
                except Exception as e:
                    print(f"Error loading {pkl_path} for vocabulary: {e}")

    vocab_map, inv_vocab_map, vocab_list = create_vocabulary(all_text)
    
    test_processed = create_processed_dataframe(test_df, target_column, vocab_map)
    
    return test_processed, vocab_map, inv_vocab_map, vocab_list



def apply_repetition_penalty(logits, input_ids, repetition_penalty, vocab_info):
    if repetition_penalty == 1.0:
        return logits
    
    batch_size, vocab_size = logits.shape
    
    special_tokens = {
        vocab_info['vocab_map'].get('<pad>', 0),
        vocab_info['vocab_map'].get('<bos>', 2), 
        vocab_info['vocab_map'].get('<eos>', 3),
        vocab_info['vocab_map'].get('<unk>', 1)
    }
    
    for batch_idx in range(batch_size):
        generated_tokens = input_ids[batch_idx, 1:].tolist()
        
        token_counts = {}
        for token in generated_tokens:
            if token not in special_tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
        
        for token, count in token_counts.items():
            if token < vocab_size:
                penalty = repetition_penalty ** count
                
                if logits[batch_idx, token] > 0:
                    logits[batch_idx, token] = logits[batch_idx, token] / penalty
                else:
                    logits[batch_idx, token] = logits[batch_idx, token] * penalty
    
    return logits

def generate_autoregressive(model, pose_values, vocab_info, device, 
                           max_length=12, temperature=0.7, repetition_penalty=1.2):
    model.eval()
    
    bos_token = vocab_info['vocab_map'].get('<bos>', 2)
    eos_token = vocab_info['vocab_map'].get('<eos>', 3)
    
    generated_ids = torch.tensor([[bos_token]], dtype=torch.long, device=device)
    
    with torch.no_grad():
        for step in range(max_length - 1):
            attention_mask = torch.ones_like(generated_ids)
            
            inputs = {
                'pose_values': pose_values,
                'input_ids': generated_ids,
                'attention_mask': attention_mask,
                'use_cache': False
            }
            
            outputs = model(**inputs)
            
            pose_seq_len = pose_values.size(1)
            current_text_len = generated_ids.size(1)
            next_token_logits = outputs.logits[:, pose_seq_len + current_text_len - 1, :].clone()
            
            if repetition_penalty != 1.0:
                next_token_logits = apply_repetition_penalty(
                    next_token_logits, generated_ids, repetition_penalty, vocab_info
                )
            
            next_token_logits = next_token_logits / temperature
            
            if temperature > 0:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            if next_token.item() == eos_token:
                break
                
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
    
    return generated_ids

