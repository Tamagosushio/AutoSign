# utils/progressive_dataset.py
import torch
import pandas as pd
from utils.datasetv2 import PoseDatasetV2

class ProgressivePoseDataset(PoseDatasetV2):
    def __init__(self, dataset_name, label_csv, split_type, target_enc_df, vocab_map, 
                 transform=None, augmentations=True, augmentations_prob=0.5, additional_joints=True):
        super().__init__(dataset_name, label_csv, split_type, target_enc_df, transform, 
                        augmentations, augmentations_prob, additional_joints)
        
        self.vocab_map = vocab_map
        self.bos_token = vocab_map.get('<BOS>', 1)
        self.eos_token = vocab_map.get('<EOS>', 2) 
        self.pad_token = vocab_map.get('<PAD>', 0)
        
        # Store raw text for decoder targets
        self.raw_texts = []
        all_data = pd.read_csv(label_csv, delimiter="|")
        
        for _, row in all_data.iterrows():
            sample_id = str(row["id"])
            if sample_id in [f for f in self.files]:
                self.raw_texts.append(row["gloss"])

    def create_decoder_targets(self, text):
        # Convert text to token sequence with BOS/EOS
        tokens = text.split()
        token_ids = [self.bos_token] + [self.vocab_map.get(token, self.pad_token) for token in tokens] + [self.eos_token]
        return torch.tensor(token_ids, dtype=torch.long)

    def __getitem__(self, idx):
        file_path, pose, ctc_label = super().__getitem__(idx)
        
        # Get raw text for this sample
        raw_text = self.raw_texts[idx]
        
        # Create decoder targets
        decoder_targets = self.create_decoder_targets(raw_text)
        
        return file_path, pose, ctc_label, decoder_targets, raw_text