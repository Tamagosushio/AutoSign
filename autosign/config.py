from typing import Optional, Union, Tuple, List, Literal, Dict, Any


class AutoSignConfig:
    """
    Configuration class for AutoSign model.
    This class holds all the hyperparameters and settings for the model,
    including GPT-2 and ViT configurations, augmentation settings, and more.
    """
    def __init__(
        self,
        gpt2_hf_model: str = 'aubmindlab/aragpt2-base',
        # vit_hf_model: str = 'google/vit-base-patch16-224',
        vocab_size: Optional[int] = 50257,
        max_position_embeddings: Optional[int] = 1200,
        hidden_size: Optional[int] = 768,
        num_hidden_layers: Optional[int] = 12,
        num_attention_heads: Optional[int] = 12,
        patch_size: Optional[Union[Tuple[int], List[int]]] = (4, 8),      # (height, width)
        image_size: Optional[Union[Tuple[int], List[int]]] = (32, 128),   # (height, width)
        num_channels: Optional[int] = 3,
        resid_pdrop: Optional[float] = 0.1,
        embd_pdrop: Optional[float] = 0.1,
        attn_pdrop: Optional[float] = 0.1,
        layer_norm_epsilon: Optional[float] = 1e-5,
        attn_implementation: Literal['eager', 'flash_attention_2'] = 'eager',
        input_dim: Optional[int] = 86,
        d_model: Optional[int] = 512,
        pose_embedding_length: Optional[int] = 100,  
        augmentation_config: Optional[Union[str, Dict[str, Any]]] = 'moderate',
        rotation_angle: Optional[float] = 0.0,  # Default no rotation
        pose_data_path: str = "data/pose_data_isharah1000_hands_lips_body_May12.pkl",
        pose_dropout: Optional[float] = 0.1,
        include_face: bool = False,
        use_scheduler: bool = True,
        use_1dcnn: bool = True, 
        cnn_layers: int = 2,   
        exclude_body: bool = False, 
    ):
        self.gpt2_hf_model = gpt2_hf_model
        # self.vit_hf_model = vit_hf_model
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self._attn_implementation = attn_implementation
        # self.input_dim = input_dim  # 86 joints
        self.d_model = d_model
        self.pose_embedding_length = pose_embedding_length
        self.rotation_angle = rotation_angle  
        self.pose_dropout = pose_dropout
        self.include_face = include_face
        self.use_scheduler = use_scheduler
        self.use_1dcnn = use_1dcnn
        self.cnn_layers = cnn_layers
        self.exclude_body = exclude_body

        print(f"Using num_hidden_layers: {self.num_hidden_layers}")
        if self.use_1dcnn and self.cnn_layers not in [2, 3]:
            raise ValueError(f"cnn_layers must be 2 or 3, got {self.cnn_layers}")

        if self.exclude_body and not self.include_face:
            print("Excluding body, excluding face. Using hands only.")
            # Hands only: rh + lh = 21 + 21 = 42 joints
            self.input_dim = 42
            
        elif self.exclude_body and self.include_face:
            print("[CONFIG] Excluding body, including face. Using hands + face joints.")
            # Hands + Face (lips): rh + lh + lips = 21 + 21 + 19 = 61 joints
            self.input_dim = 61
            
        elif not self.exclude_body and not self.include_face:
            print("[CONFIG] Including body, excluding face. Using body + hands joints.")
            # Body + Hands (no face): rh + lh + body = 21 + 21 + 25 = 67 joints
            self.input_dim = 67
            
        elif not self.exclude_body and self.include_face:
            print("Including body and face. Using body + hands + face joints.")
            # Full: rh + lh + lips + body = 21 + 21 + 19 + 25 = 86 joints
            self.input_dim = input_dim  
            
        else:
            self.input_dim = input_dim

        
            
        if isinstance(augmentation_config, str):
            self.augmentation_config = AUGMENTATION_CONFIGS.get(augmentation_config, AUGMENTATION_CONFIGS['moderate'])
        else:
            self.augmentation_config = augmentation_config or AUGMENTATION_CONFIGS['moderate']

        self.n_inner = None
        self.scale_attn_weights = True
        self.scale_attn_by_inverse_layer_idx = False
        self.reorder_and_upcast_attn = False
        self.add_cross_attention = False
        self.activation_function = "gelu_new"


AUGMENTATION_CONFIGS = {
    'minimal': {
        'use_jitter': True,
        'use_scale': True,
        'use_dropout': False,
        'use_time_warp': False,
        'use_frame_dropout': False,
        'use_speed_change': True,
        'use_sequence_masking': False,
        'frame_dropout_prob': 0.05,
        'speed_change_prob': 0.4,
        'jitter_std': 0.005, 
        'scale_range': (0.9, 1.1), 
    },
    'moderate': {
        'use_jitter': True,
        'use_scale': True,
        'use_dropout': True,
        'use_time_warp': True,
        'use_frame_dropout': True,
        'use_speed_change': True,
        'use_sequence_masking': True,
        'frame_dropout_prob': 0.05,
        'speed_change_prob': 0.4,
        'sequence_masking_prob': 0.15,
        'jitter_std': 0.01,
        'scale_range': (0.85, 1.15),
        'time_warp_shift': 1, 
    },
    'aggressive': {
        'use_jitter': True,
        'use_scale': True,
        'use_dropout': True,
        'use_time_warp': True,
        'use_frame_dropout': True,
        'use_speed_change': True,
        'use_sequence_masking': True,
        'frame_dropout_prob': 0.1,
        'speed_change_prob': 0.7,
        'sequence_masking_prob': 0.2,
        'jitter_std': 0.015,
        'scale_range': (0.8, 1.2),
        'time_warp_shift': 2,
    }
}