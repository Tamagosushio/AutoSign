import torch
import numpy as np
from transformers import GPT2Tokenizer
from typing import List, Union, Optional
from config import AutoSignConfig
from data import AutoSignProcessorOutput


class AutoSignProcessor:
    def __init__(self, config: AutoSignConfig, add_bos_token: bool = False, add_eos_token: bool = False):
        
        self.tokeniser = GPT2Tokenizer.from_pretrained(
            config.gpt2_hf_model,
            add_bos_token=add_bos_token,
            model_max_length=config.max_position_embeddings - config.pose_embedding_length
        )
        self.tokeniser.pad_token = self.tokeniser.bos_token
        self.tokeniser.add_eos_token = add_eos_token

        self.config = config
        self.pose_embedding_length = config.pose_embedding_length

        self.tokeniser.build_inputs_with_special_tokens = modified_build_inputs_with_special_tokens.__get__(
            self.tokeniser
        )

    def __call__(
        self,
        poses: Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]] = None,
        texts: Union[str, List[str]] = None,
        return_labels: bool = False,
        padding: Union[bool, str] = False,
        max_sequence_length: Optional[int] = None,
        *args,
        **kwargs
    ) -> AutoSignProcessorOutput:
        
        text_inputs = self.tokeniser(
            texts, padding=padding, *args, **kwargs
        ) if texts is not None else None

        pose_inputs = self._process_poses(
            poses, max_sequence_length=max_sequence_length
        ) if poses is not None else None

        return AutoSignProcessorOutput(
            pose_values=pose_inputs["pose_values"] if poses is not None else None,
            input_ids=text_inputs['input_ids'] if texts is not None else None,
            attention_mask=text_inputs['attention_mask'] if texts is not None else None,
            labels=text_inputs['input_ids'] if texts is not None and return_labels else None
        )

    def _process_poses(
        self, 
        poses: Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]],
        max_sequence_length: Optional[int] = None
    ) -> dict:
        """
        Process pose data for the model
        
        Args:
            poses: Pose data in various formats
            max_sequence_length: Maximum sequence length for padding
            
        Returns:
            Dictionary with processed pose values
        """
        
        if isinstance(poses, (np.ndarray, torch.Tensor)):
            if len(poses.shape) == 2:  # (seq_len, 172)
                pose_list = [poses]
            elif len(poses.shape) == 3:  # (batch_size, seq_len, 172)
                pose_list = [poses[i] for i in range(poses.shape[0])]
            else:
                raise ValueError(f"Unexpected pose shape: {poses.shape}")
        elif isinstance(poses, list):
            pose_list = poses
        else:
            raise ValueError(f"Unsupported pose type: {type(poses)}")
        
        pose_tensors = []
        sequence_lengths = []
        
        for pose_seq in pose_list:
            if isinstance(pose_seq, np.ndarray):
                pose_seq = torch.from_numpy(pose_seq).float()
            elif not isinstance(pose_seq, torch.Tensor):
                pose_seq = torch.tensor(pose_seq, dtype=torch.float32)
            
            # correct shape: (seq_len, 172)
            if len(pose_seq.shape) == 1:
                pose_seq = pose_seq.unsqueeze(0)
            elif len(pose_seq.shape) == 3 and pose_seq.shape[-1] == 2:
                pose_seq = pose_seq.reshape(pose_seq.shape[0], -1)
            
            if pose_seq.shape[-1] != self.config.input_dim * 2:
                raise ValueError(
                    f"Expected pose feature dimension {self.config.input_dim * 2}, "
                    f"got {pose_seq.shape[-1]}"
                )
            
            pose_tensors.append(pose_seq)
            sequence_lengths.append(pose_seq.shape[0])
        
        if max_sequence_length is None:
            max_sequence_length = max(sequence_lengths)
        
        # Pad sequences
        batch_size = len(pose_tensors)
        feature_dim = pose_tensors[0].shape[-1]
        
        padded_poses = torch.zeros(batch_size, max_sequence_length, feature_dim)
        
        for i, pose_seq in enumerate(pose_tensors):
            seq_len = min(pose_seq.shape[0], max_sequence_length)
            padded_poses[i, :seq_len] = pose_seq[:seq_len]
        
        return {
            "pose_values": padded_poses,
            "sequence_lengths": sequence_lengths
        }

    def preprocess_pose_sequence(
        self, 
        pose_sequence: Union[np.ndarray, torch.Tensor],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Preprocess a single pose sequence
        
        Args:
            pose_sequence: Raw pose data
            normalize: Whether to normalize coordinates
            
        Returns:
            Processed pose tensor
        """
        if isinstance(pose_sequence, np.ndarray):
            pose_sequence = torch.from_numpy(pose_sequence).float()
        
        if len(pose_sequence.shape) == 3:  # (seq_len, 86, 2)
            pose_sequence = pose_sequence.reshape(pose_sequence.shape[0], -1)
        elif len(pose_sequence.shape) == 2 and pose_sequence.shape[-1] != 172:
            #  single frame
            if pose_sequence.shape[0] == 86 and pose_sequence.shape[1] == 2:
                pose_sequence = pose_sequence.reshape(1, -1)  # (1, 172)
            elif pose_sequence.shape[1] == 86 and pose_sequence.shape[0] == 2:
                pose_sequence = pose_sequence.T.reshape(1, -1)  # (1, 172)
        

        
        return pose_sequence

    def batch_poses(
        self, 
        pose_list: List[Union[np.ndarray, torch.Tensor]], 
        max_length: Optional[int] = None
    ) -> AutoSignProcessorOutput:
        """
        Batch multiple pose sequences with padding
        """
        return self(poses=pose_list, max_sequence_length=max_length)


def modified_build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    if self.add_bos_token:
        bos_token_ids = [self.bos_token_id]
    else:
        bos_token_ids = []

    if self.add_eos_token:
        eos_token_ids = [self.eos_token_id]
    else:
        eos_token_ids = []

    output = bos_token_ids + token_ids_0 + eos_token_ids

    if token_ids_1 is None:
        return output

    return output + bos_token_ids + token_ids_1

