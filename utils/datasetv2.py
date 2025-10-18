import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import pandas as pd
import math
import random
from random import randrange
import cv2
import pickle
from autosign.config import AUGMENTATION_CONFIGS




import pandas as pd
import torch
from collections import Counter
from typing import Dict, List, Tuple, Union

def load_and_process_text_data(train_csv: str, dev_csv: str, target_column: str = 'text'):
    """
    Load and process text data for pose-to-text training
    
    Args:
        train_csv: Path to training CSV file
        dev_csv: Path to development CSV file  
        target_column: Which column to use as target ('text' or 'gloss')
    
    Returns:
        Tuple of processed dataframes and vocabulary mappings
    """
    
    # Load data
    train_df = pd.read_csv(train_csv, delimiter="|")
    dev_df = pd.read_csv(dev_csv, delimiter="|")
    
    train_df = train_df.dropna(subset=['id', target_column])
    dev_df = dev_df.dropna(subset=['id', target_column])
    
    print(f"Loaded {len(train_df)} training samples, {len(dev_df)} dev samples")
    print(f"Using target column: {target_column}")
    
    all_text = train_df[target_column].tolist()
    vocab_map, inv_vocab_map, vocab_list = create_vocabulary(all_text)
    
    print(f"Vocabulary size: {len(vocab_list)}")
    print(f"Sample texts:")
    for i, text in enumerate(all_text[:3]):
        print(f"  {i+1}: {text}")
    
    train_processed = create_processed_dataframe(train_df, target_column)
    dev_processed = create_processed_dataframe(dev_df, target_column)
    
    return train_processed, dev_processed, vocab_map, inv_vocab_map, vocab_list

def create_vocabulary(texts: List[str], min_freq: int = 1):
    """
    Create vocabulary from text data
    
    Args:
        texts: List of text strings
        min_freq: Minimum frequency for a word to be included
        
    Returns:
        vocab_map, inv_vocab_map, vocab_list
    """
    
    word_counts = Counter()
    for text in texts:
        words = text.strip().split()
        word_counts.update(words)
    
    vocab_words = [word for word, count in word_counts.items() if count >= min_freq]
    vocab_words = sorted(vocab_words)  
    
    # Add special tokens
    special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']
    vocab_list = special_tokens + vocab_words
    
    # Create mappings
    vocab_map = {word: idx for idx, word in enumerate(vocab_list)}
    inv_vocab_map = {idx: word for idx, word in enumerate(vocab_list)}
    
    print(f"Created vocabulary with {len(vocab_list)} tokens")
    print(f"Special tokens: {special_tokens}")
    print(f"Sample vocab words: {vocab_words[:10]}")
    
    return vocab_map, inv_vocab_map, vocab_list

def create_processed_dataframe(df: pd.DataFrame, target_column: str):
    """
    Create processed dataframe with text column for easy access
    """
    processed_df = df.copy()
    processed_df['target_text'] = processed_df[target_column]
    return processed_df

def convert_labels_to_text(labels: Union[torch.Tensor, List], inv_vocab_map: Dict[int, str] = None, 
                          processed_df: pd.DataFrame = None, batch_indices: List[int] = None):
    """
    Convert labels to text strings
    
    Args:
        labels: Either tokenized labels 
        inv_vocab_map: Inverse vocabulary mapping
        processed_df: Processed dataframe with text
        batch_indices: Indices in the dataset for this batch
        
    Returns:
        List of text strings
    """
    
    if isinstance(labels, torch.Tensor):
        if inv_vocab_map is None:
            raise ValueError("inv_vocab_map required for tokenized labels")
            
        text_strings = []
        for label_sequence in labels:
            # Convert tensor to list and decode
            token_ids = label_sequence.cpu().numpy().tolist()
            words = []
            for token_id in token_ids:
                if token_id in inv_vocab_map:
                    word = inv_vocab_map[token_id]
                    if word not in ['<pad>', '<bos>', '<eos>']: 
                        words.append(word)
                else:
                    words.append('<unk>')
            text_strings.append(' '.join(words))
        return text_strings
        
    elif isinstance(labels, list) and len(labels) > 0 and isinstance(labels[0], str):
        return labels
        
    elif processed_df is not None and batch_indices is not None:
        text_strings = []
        for idx in batch_indices:
            if idx < len(processed_df):
                text_strings.append(processed_df.iloc[idx]['target_text'])
            else:
                text_strings.append('')
        return text_strings
        
    else:
        raise ValueError("Cannot convert labels to text - insufficient information provided")

def encode_text_to_tokens(text: str, vocab_map: Dict[str, int], max_length: int = None):
    """
    Encode text string to token IDs
    
    Args:
        text: Input text string
        vocab_map: Vocabulary mapping 
        max_length: Maximum sequence length
        
    Returns:
        List of token IDs
    """
    words = text.strip().split()
    
    token_ids = [vocab_map.get('<bos>', 2)] 
    for word in words:
        token_id = vocab_map.get(word, vocab_map.get('<unk>', 1))
        token_ids.append(token_id)
    token_ids.append(vocab_map.get('<eos>', 3))  
    
    if max_length:
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            token_ids[-1] = vocab_map.get('<eos>', 3)  
        else:
            pad_token = vocab_map.get('<pad>', 0)
            token_ids.extend([pad_token] * (max_length - len(token_ids)))
    
    return token_ids


    
lipsUpperOuter = [61, 185, 40, 39, 37, 0, 267, 269, 270, 291]
lipsLowerOuter = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
lips = sorted(set(lipsUpperOuter + lipsLowerOuter))  # Remove duplicates and sort
NUM_LIPS = 19  # after deduplication

        
class PoseDatasetV2(Dataset):

    def rotate(self, origin, point, angle):
        """ Rotates a point around an origin. """
        ox, oy = origin
        px, py = point
        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy

    def augment_jitter(self, keypoints, std_dev=0.01):
        """ Adds Gaussian noise to keypoints. """
        noise = np.random.normal(loc=0, scale=std_dev, size=keypoints.shape)
        return keypoints + noise

    def augment_time_warp(self, pose_data, max_shift=2):
        """ Randomly shifts frames to simulate varying signing speed. """
        T = pose_data.shape[0]
        new_data = np.zeros_like(pose_data)
        for i in range(T):
            shift = np.random.randint(-max_shift, max_shift + 1)
            new_idx = np.clip(i + shift, 0, T - 1)
            new_data[i] = pose_data[new_idx]
        return new_data

    def augment_dropout(self, keypoints, drop_prob=0.1):
        """ Randomly drops some keypoints. """
        mask = np.random.rand(*keypoints.shape[:1]) > drop_prob
        keypoints *= mask[:, np.newaxis]
        return keypoints

    def augment_scale(self, keypoints, scale_range=(0.8, 1.2)):
        """ Randomly scales keypoints."""
        scale = np.random.uniform(*scale_range)
        return keypoints * [scale, scale]

    def augment_frame_dropout(self, pose_data, drop_prob=0.1):
        """ Randomly drops full frames. """
        T = pose_data.shape[0]
        mask = np.random.rand(T) > drop_prob
        return pose_data * mask[:, np.newaxis, np.newaxis]

    def augment_sequence_masking(self, pose_data, mask_prob=0.15, mask_length_ratio=0.1):
        """NEW: Mask consecutive frames to force the model to use context"""
        T = pose_data.shape[0]
        if T < 10:
            return pose_data
        
        if np.random.rand() < mask_prob:
            mask_length = max(1, int(T * mask_length_ratio))
            start_idx = np.random.randint(0, max(1, T - mask_length))
            mask_end = min(start_idx + mask_length, T)
            
            if start_idx > 0 and mask_end < T:
                # Linear interpolation between boundaries
                for i in range(start_idx, mask_end):
                    alpha = (i - start_idx) / mask_length
                    pose_data[i] = (1 - alpha) * pose_data[start_idx - 1] + alpha * pose_data[mask_end]
            else:
                noise = np.random.normal(0, 0.02, pose_data[start_idx:mask_end].shape)
                pose_data[start_idx:mask_end] += noise
        
        return pose_data

    def augment_data(self, data, angle):
        """Updated to use configuration"""
        if np.random.rand() < 0.5:
            data = np.array([self.rotate((0.5, 0.5), frame, angle) for frame in data])
        if np.random.rand() < 0.5:
            data = self.augment_jitter(data)
        if np.random.rand() < 0.5:
            data = self.augment_scale(data)
        if np.random.rand() < 0.5:
            data = self.augment_dropout(data)
        return data

    def normalize(self, pose):
        pose[:,:] -= pose[0]  
        pose[:,:] -= np.min(pose, axis=0)
        
        max_vals = np.max(pose, axis=0)
        pose[:,:] /= max(max_vals)

        pose[:,:] = pose[:,:] - np.mean(pose[:,:])
        pose[:,:] = pose[:,:] / np.max(np.abs(pose[:,:]))
        pose[:,:] = pose[:,:] * 0.5

        return pose

    def normalize_face(self, pose):
        #set coordinate frame as the lip
        pose[:,:] -= pose[0]  
        pose[:,:] -= np.min(pose, axis=0)
        
        #scale them to a box of 1x1
        max_vals = np.max(pose, axis=0)
        pose[:,:] /= max(max_vals)

        # Subtract the mean from each element and divide by the maximum absolute value
        # The values are then [-0.5,0.5] spread over zero 
        pose[:,:] = pose[:,:] - np.mean(pose[:,:])
        pose[:,:] = pose[:,:] / np.max(np.abs(pose[:,:]))
        pose[:,:] = pose[:,:] * 0.5

        return pose

    def normalize_body(self, pose ):
        #set coordinate frame as the neck
        pose[:,:] -= pose[0]
        pose[:,:] -= np.min(pose, axis=0)
        
        #scale them to a box of 1x1
        max_vals = np.max(pose, axis=0)
        pose[:,:] /= max(max_vals)

        # Subtract the mean from each element and divide by the maximum absolute value
        # The values are then [-0.5,0.5] spread over zero 
        pose[:,:] = pose[:,:] - np.mean(pose[:,:])
        pose[:,:] = pose[:,:] / np.max(np.abs(pose[:,:]))
        pose[:,:] = pose[:,:] * 0.5

        return pose

    def augment_realistic_speed_change(self, pose_data, speed_range=(0.75, 1.25)):
        """
        #1 MOST IMPORTANT: Realistic speed variations
        This addresses the biggest problem in sign language recognition.
        """
        speed_factor = np.random.uniform(*speed_range)
        original_length = pose_data.shape[0]
        new_length = max(16, int(original_length * speed_factor)) 
        
        old_indices = np.linspace(0, original_length - 1, new_length)
        new_pose_data = np.zeros((new_length, *pose_data.shape[1:]))
        
        for i, old_idx in enumerate(old_indices):
            if old_idx == int(old_idx):
                # Exact frame match
                new_pose_data[i] = pose_data[int(old_idx)]
            else:
                # Linear interpolation between frames
                low_idx = int(np.floor(old_idx))
                high_idx = min(int(np.ceil(old_idx)), original_length - 1)
                alpha = old_idx - low_idx
                new_pose_data[i] = (1 - alpha) * pose_data[low_idx] + alpha * pose_data[high_idx]
        
        return new_pose_data

    def __init__(self, dataset_name2, label_csv, split_type, target_enc_df, transform=None, 
                augmentations=True, augmentations_prob=0.5, additional_joints=True, 
                augmentation_config='moderate', pose_data_path=None, include_face=False, exclude_body=False):

        self.dataset_name = dataset_name2
        self.split_type = split_type 
        self.transform = transform
        self.augmentations = augmentations
        self.augmentations_prob = augmentations_prob
        self.additional_joints = additional_joints
        self.include_face=include_face
        self.exclude_body = exclude_body    
        
        if isinstance(augmentation_config, str):
            self.aug_config = AUGMENTATION_CONFIGS.get(augmentation_config, AUGMENTATION_CONFIGS['moderate'])
        else:
            self.aug_config = augmentation_config
        
        print(f"Using augmentation config: {augmentation_config}")
        
        if pose_data_path is None:
            pose_data_path = "data/pose_data_isharah1000_hands_lips_body_May12.pkl"
        
        with open(pose_data_path, 'rb') as f:
            print(f"Loading pose data from {pose_data_path}")
            self.pose_dict = pickle.load(f)




        self.files = []
        self.labels = []

        self.all_data = pd.read_csv(label_csv, delimiter="|")
        # if "isharah" in self.dataset_name:
        #     self.all_data = self.all_data[self.all_data["id"].notna()]
        #     self.all_data = self.all_data[self.all_data["gloss"].notna()]
        # elif "iam" in self.dataset_name:
        #     self.all_data = self.all_data[self.all_data["id"].notna()]
        #     self.all_data = self.all_data[self.all_data["text"].notna()]

        for _, row in self.all_data.iterrows():
            sample_id = row["id"]
            enc_label = target_enc_df[target_enc_df["id"] == sample_id]["enc"]
            if enc_label.empty:
                print(f"Warning: No encoded label found for sample ID {sample_id}. Skipping.")

            if not enc_label.empty and sample_id in self.pose_dict.keys():
                self.files.append(sample_id)
                self.labels.append(enc_label.iloc[0])

        print(f"Loaded {len(self.files)} samples for split: {split_type}")



    def readPose(self, sample_id):
        """Reads pose data for a given sample ID.

        Args:
            sample_id (str): The ID of the sample to read.
        Returns:
            np.ndarray: The pose data for the sample, shape (T, J, D).
        Raises:
            ValueError: If pose data is not found or is empty.
        """
        pose_data = self.pose_dict[sample_id]['keypoints']
        # print(f"Loading pose data for {sample_id}, shape: {pose_data.shape}")

        if pose_data is None or pose_data.shape[0] == 0:
            raise ValueError(f"Error loading pose data for {sample_id}")

        T, J, D = pose_data.shape
        aug = False

        if self.augmentations and np.random.rand() < self.augmentations_prob:
            aug = True
            
            angle = np.radians(np.random.uniform(-13, 13))
            pose_data = self.augment_time_warp(pose_data)
            pose_data = self.augment_frame_dropout(pose_data)


        right_hand = pose_data[:, 0:21, :2]
        left_hand = pose_data[:, 21:42, :2]
        lips = pose_data[:, 42:42+NUM_LIPS, :2]
        body = pose_data[:,42+NUM_LIPS:]

        right_joints, left_joints, face_joints, body_joints = [], [], [], []

        for ii in range(T):

            rh = right_hand[ii]
            lh = left_hand[ii]
            fc = lips[ii]
            bd = body[ii]

            if rh.sum() == 0:
                rh[:] = right_joints[-1] if ii != 0 else np.zeros((21, 2))
            else:
                if aug:
                    rh = self.augment_data(rh, angle)
                rh = self.normalize(rh)

            if lh.sum() == 0:
                lh[:] = left_joints[-1] if ii != 0 else np.zeros((21, 2))
            else:
                if aug:
                    lh = self.augment_data(lh, angle)
                lh = self.normalize(lh)

            if fc.sum() == 0:
                fc[:] = face_joints[-1] if ii != 0 else np.zeros((len(fc), 2))
            else:
                fc = self.normalize_face(fc)
            
            if bd.sum() == 0:
                bd[:] = body_joints[-1] if ii != 0 else np.zeros((len(bd), 2))
            else:
                bd = self.normalize_body(bd)

            right_joints.append(rh)
            left_joints.append(lh)
            face_joints.append(fc)
            body_joints.append(bd)

        for ljoint_idx in range(len(left_joints) - 2, -1, -1):
            if left_joints[ljoint_idx].sum() == 0:
                left_joints[ljoint_idx] = left_joints[ljoint_idx + 1].copy()

        for rjoint_idx in range(len(right_joints) - 2, -1, -1):
            if right_joints[rjoint_idx].sum() == 0:
                right_joints[rjoint_idx] = right_joints[rjoint_idx + 1].copy()

            concatenated_joints = np.concatenate((right_joints, left_joints), axis=1)
            
            if self.additional_joints:
                # Add face/lips if include_face is True
                if getattr(self, 'include_face', True):
                    concatenated_joints = np.concatenate((concatenated_joints, face_joints), axis=1)
                
                # Add body if exclude_body is False
                if not getattr(self, 'exclude_body', False):
                    concatenated_joints = np.concatenate((concatenated_joints, body_joints), axis=1)
            

            
            return concatenated_joints
        


    def __len__(self):
        return len(self.files)
    
    def get_file_path(self, idx):
        return self.files[idx]
    
    def pad_or_crop_sequence(self, sequence, min_len=32, max_len=1000):
        T, J, D = sequence.shape
        if T < min_len:
            pad_len = min_len - T
            pad = np.zeros((pad_len, J, D))
            sequence = np.concatenate((sequence, pad), axis=0)
            T = sequence.shape[0]  # update T
        if sequence.shape[0] > max_len:
            sequence = sequence[:max_len]
        
        return sequence


    def __getitem__(self, idx):
        sample_id = self.files[idx]  
        file_path = str(sample_id) 
        
        pose = self.readPose(sample_id) 
        pose = self.pad_or_crop_sequence(pose, min_len=32, max_len=1000)
        pose = torch.from_numpy(pose).float()

        if self.transform:
            pose = self.transform(pose)

        label = self.labels[idx]
        label_tensor = torch.as_tensor(label, dtype=torch.long)
        
        # Flatten pose data: 
        T, J, D = pose.shape
        pose_flattened = pose.view(T, J * D)  
        
        seq_len = len(label_tensor)
        attention_mask = torch.ones(seq_len, dtype=torch.long)
        
        return {
            'file_path': file_path, 
            'pose_values': pose_flattened, 
            'input_ids': label_tensor,   
            'attention_mask': attention_mask, 
            'labels': label_tensor.clone()   
        }

        
    