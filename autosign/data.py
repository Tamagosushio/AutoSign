import torch
import numpy as np

from dataclasses import dataclass
from typing import Optional, Union, List


@dataclass
class AutoSignModelOutput:
    hidden_states: torch.FloatTensor
    past_key_values: Optional[torch.FloatTensor] = None


@dataclass
class AutoSignLMHeadModelOutput:
    logits: torch.FloatTensor
    loss: Optional[torch.FloatTensor] = None
    accuracy: Optional[torch.FloatTensor] = None
    past_key_values: Optional[torch.FloatTensor] = None


@dataclass
class AutoSignProcessorOutput:
    pose_values: Optional[torch.FloatTensor] = None  # Shape: (batch_size, sequence_length, 172)
    input_ids: Optional[Union[torch.LongTensor, np.ndarray, List[int]]] = None
    attention_mask: Optional[Union[torch.FloatTensor, np.ndarray, List[int]]] = None
    labels: Optional[Union[torch.LongTensor, np.ndarray, List[int]]] = None
    


@dataclass
class PoseSequence:
    """
    Represents a sequence of poses for sign language/gesture recognition
    """
    poses: torch.FloatTensor  # Shape: (sequence_length, 86, 2)
    sequence_length: int     
    text: Optional[str] = None   
    tokens: Optional[List[int]] = None 


@dataclass
class AutoSignBatch:
    """
    Represents a batch of pose sequences with proper padding and attention masks
    """
    pose_values: torch.FloatTensor   
    input_ids: torch.LongTensor        
    attention_mask: torch.FloatTensor  
    pose_attention_mask: torch.FloatTensor  
    labels: Optional[torch.LongTensor] = None
    sequence_lengths: Optional[List[int]] = None 


