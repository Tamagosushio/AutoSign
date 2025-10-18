import os
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence
from utils.datasetv2 import PoseDatasetV2
from utils.utils import GaussianNoise, send_inputs_to_device, decode_predictions, load_and_process_text_data
import argparse
import shutil

def custom_collate_fn(batch):
    """Same custom collate function as training"""
    file_paths = [item['file_path'] for item in batch]
    pose_values = [item['pose_values'] for item in batch]
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    pose_values_padded = pad_sequence(pose_values, batch_first=True, padding_value=0.0)
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return {
        'file_path': file_paths,
        'pose_values': pose_values_padded,
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded,
        'labels': labels_padded
    }

def create_processed_dataframe_for_test(df: pd.DataFrame, target_column: str, vocab_map):
    """Create processed dataframe with encoding for test data"""
    from utils.utils import encode_text_to_tokens
    
    processed_df = df.copy()
    
    encoded_texts = []
    for _, row in processed_df.iterrows():
        text = row.get(target_column, '')
        
        # Handle missing gloss values (common in test data)
        if pd.isna(text) or text is None or text == '':
            # Create dummy encoding for test data - just BOS + EOS
            print(f"Warning: Missing gloss for test sample {row.get('id', 'unknown')}")
            encoded = [vocab_map.get('<bos>', 2), vocab_map.get('<eos>', 3)]
        else:
            # print(f"Encoding text: {text}")
            encoded = encode_text_to_tokens(str(text).strip(), vocab_map)
        
        encoded_texts.append(encoded)
    
    processed_df['enc'] = encoded_texts
    return processed_df

def setup_test_data(args):
    """Setup test data using the SAME process as training"""
    train_csv = f"annotations_v2/{args.mode}/train.txt"
    dev_csv = f"annotations_v2/{args.mode}/dev.txt"
    test_csv = f"annotations_v2/{args.mode}/{args.mode}_test.txt"
    
    train_processed, dev_processed, vocab_map, inv_vocab_map, vocab_list = load_and_process_text_data(
        train_csv, dev_csv, target_column='gloss'
    )
    
    print(f"[DATA] Vocabulary loaded with {len(vocab_list)} tokens")
    print(f"[DATA] Sample vocab: {list(vocab_map.keys())[:10]}")
    
    # Load test data
    test_df = pd.read_csv(test_csv, delimiter="|")
    test_df = test_df.dropna(subset=['id'])
    
    print(f"[DATA] Loaded {len(test_df)} test samples")
    
    # Process test data using the training vocabulary
    test_processed = create_processed_dataframe_for_test(test_df, 'gloss', vocab_map)
    
    print('[DATA] Test data processed successfully')
    print(f"[DATA] Sample test encodings: {test_processed['enc'].head().tolist()}")
    
    dataset_test = PoseDatasetV2(
        dataset_name2="isharah",
        label_csv=test_csv,
        split_type="test",
        target_enc_df=test_processed,
        augmentations=False,  
        pose_data_path=f"data/pose_data_isharah1000_{args.mode}_test.pkl"
    )
    
    test_loader = DataLoader(
        dataset_test, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        collate_fn=custom_collate_fn  
    )
    
    vocab_info = {
        'vocab_map': vocab_map, 
        'inv_vocab_map': inv_vocab_map, 
        'vocab_list': vocab_list,
        'vocab_size': len(vocab_list)
    }
    
    print(f"[DATA] Test DataLoader created with {len(dataset_test)} samples")
    return test_loader, vocab_info

def load_checkpoint(checkpoint_path, device):
    """Load checkpoint and return epoch info"""
    print(f"[MODEL] Loading checkpoint: {checkpoint_path}")
    # checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    epoch = checkpoint.get('epoch', 'unknown')
    train_loss = checkpoint.get('train_loss', 'unknown')
    val_loss = checkpoint.get('val_loss', 'unknown')
    val_wer = checkpoint.get('val_wer', 'unknown')
    
    print(f"[MODEL] Checkpoint info:")
    print(f"  Epoch: {epoch}")
    print(f"  Train Loss: {train_loss}")
    print(f"  Val Loss: {val_loss}")
    print(f"  Val WER: {val_wer}")
    
    return checkpoint

def remove_duplicates(text):
    """Remove consecutive duplicate words"""
    if not text or len(text.split()) < 2:
        return text
    
    words = text.split()
    result = [words[0]]
    
    for word in words[1:]:
        if word != result[-1]:
            result.append(word)
    
    return ' '.join(result)

def apply_repetition_penalty(logits, input_ids, repetition_penalty, vocab_info):
    """Apply repetition penalty to logits based on previously generated tokens"""
    if repetition_penalty == 1.0:
        return logits
    
    batch_size, vocab_size = logits.shape
    
    # Get special tokens
    special_tokens = {
        vocab_info['vocab_map'].get('<pad>', 0),
        vocab_info['vocab_map'].get('<bos>', 2), 
        vocab_info['vocab_map'].get('<eos>', 3),
        vocab_info['vocab_map'].get('<unk>', 1)
    }
    
    for batch_idx in range(batch_size):
        # Get tokens from this batch item 
        generated_tokens = input_ids[batch_idx, 1:].tolist()  # Skip BOS token
        
        # Count frequency of each token
        token_counts = {}
        for token in generated_tokens:
            if token not in special_tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
        
        # Apply penalty based on frequency
        for token, count in token_counts.items():
            if token < vocab_size:  # Valid token index
                penalty = repetition_penalty ** count
                
                if logits[batch_idx, token] > 0:
                    logits[batch_idx, token] = logits[batch_idx, token] / penalty
                else:
                    logits[batch_idx, token] = logits[batch_idx, token] * penalty
    
    return logits

def generate_autoregressive_test(model, pose_values, vocab_info, device, 
                                max_length=20, temperature=0.8, repetition_penalty=1.4):
    """Autoregressive generation for test data with repetition penalty"""
    model.eval()
    
    bos_token = vocab_info['vocab_map'].get('<bos>', 2)
    eos_token = vocab_info['vocab_map'].get('<eos>', 3)
    
    # Start with BOS token
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
            
            # next_token_logits = outputs.logits[:, pose_seq_len + current_text_len - 1, :].clone()
            next_token_logits = outputs.logits[:, -1, :].clone()

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                next_token_logits = apply_repetition_penalty(
                    next_token_logits, generated_ids, repetition_penalty, vocab_info
                )
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            if temperature > 0:
                # Sample from distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy selection
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            if next_token.item() == eos_token:
                break
                
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
    
    return generated_ids

def run_inference_autoregressive(model, test_loader, device, vocab_info, work_dir, checkpoint_epoch):
    """Run autoregressive inference on test data"""
    model.eval()
    
    predictions = []
    
    print(f"\n{'='*60}")
    print(f"Running AUTOREGRESSIVE Inference on Test Data")
    print(f"Checkpoint Epoch: {checkpoint_epoch}")
    print(f"{'='*60}")
    
    predictions_file = f"{work_dir}/BFH_test.csv"
    detailed_file = f"{work_dir}/test_detailed_epoch_{checkpoint_epoch}.txt"
    
    with open(predictions_file, "w", encoding='utf-8') as pred_file, \
         open(detailed_file, "w", encoding='utf-8') as detail_file:
        
        pred_file.write("id,gloss\n")
        detail_file.write(f"Test Autoregressive Predictions - Epoch {checkpoint_epoch}\n")
        detail_file.write("="*60 + "\n\n")
        
        with torch.no_grad():
            sample_count = 0
            
            for batch_idx, batch in enumerate(tqdm.tqdm(test_loader, desc="Testing", ncols=100)):
                
                inputs = {k: v for k, v in batch.items() if k != 'file_path'}
                inputs = send_inputs_to_device(inputs, device)
                
                pose_values = inputs['pose_values']
                file_names = batch['file_path']
                
                # Generate predictions for each sample in the batch
                for i in range(pose_values.size(0)):
                    single_pose = pose_values[i:i+1]
                    
                    # AUTOREGRESSIVE GENERATION with repetition penalty
                    generated_ids = generate_autoregressive_test(
                        model, single_pose, vocab_info, device, 
                        max_length=20, temperature=0.8, repetition_penalty=1.4
                    )
                    
                    # Decode generated sequence
                    generated_sequence = generated_ids[0].cpu().numpy()
                    
                    words = []
                    for token_id in generated_sequence:
                        if token_id in vocab_info['inv_vocab_map']:
                            word = vocab_info['inv_vocab_map'][token_id]
                            if word not in ['<pad>', '<bos>', '<eos>', '<unk>']:
                                words.append(word)
                    
                    prediction = ' '.join(words)
                    clean_prediction = remove_duplicates(prediction.strip())
                    
                    predictions.append(clean_prediction)
                    sample_count += 1
                    
                    # Write to CSV
                    pred_file.write(f"{sample_count},{clean_prediction}\n")
                    
                    # Write detailed info
                    detail_file.write(f"Sample {sample_count}: {file_names[i]}\n")
                    detail_file.write(f"Generated IDs: {generated_sequence.tolist()}\n")
                    detail_file.write(f"Prediction: {clean_prediction}\n")
                    detail_file.write("-" * 40 + "\n\n")
                    
                    # Print first few samples
                    if sample_count <= 10:
                        print(f"Sample {sample_count}: '{clean_prediction}'")
    
    print(f"\n[RESULTS] Generated {len(predictions)} predictions")
    print(f"[RESULTS] CSV saved to: {predictions_file}")
    print(f"[RESULTS] Detailed output saved to: {detailed_file}")
    
    # Show sample predictions
    print(f"\nSample predictions:")
    for i, pred in enumerate(predictions[:20]):
        print(f"{i+1:2d}: {pred}")
    
    return predictions

def run_inference_teacher_forcing(model, test_loader, device, vocab_info, work_dir, checkpoint_epoch):
    """Run teacher forcing inference on test data (for comparison)"""
    model.eval()
    
    predictions = []
    
    print(f"\n{'='*60}")
    print(f"Running TEACHER FORCING Inference on Test Data")
    print(f"Checkpoint Epoch: {checkpoint_epoch}")
    print(f"{'='*60}")
    
    predictions_file = f"{work_dir}/test_teacher_forcing.csv"
    
    with open(predictions_file, "w", encoding='utf-8') as pred_file:
        pred_file.write("id,gloss\n")
        
        with torch.no_grad():
            sample_count = 0
            
            for batch_idx, batch in enumerate(tqdm.tqdm(test_loader, desc="Testing TF", ncols=100)):
                
                inputs = {k: v for k, v in batch.items() if k != 'file_path'}
                inputs = send_inputs_to_device(inputs, device)
                
                # TEACHER FORCING METHOD
                outputs = model(**inputs)
                pose_seq_len = inputs['pose_values'].size(1)
                text_logits = outputs.logits[:, pose_seq_len:, :]
                batch_predictions = decode_predictions(text_logits, vocab_info)
                
                file_names = batch['file_path']
                for i, (file_name, prediction) in enumerate(zip(file_names, batch_predictions)):
                    clean_prediction = remove_duplicates(prediction.strip())
                    predictions.append(clean_prediction)
                    sample_count += 1
                    
                    pred_file.write(f"{sample_count},{clean_prediction}\n")
                    
                    if sample_count <= 10:
                        print(f"Sample {sample_count}: '{clean_prediction}'")
    
    print(f"\n[TEACHER FORCING RESULTS] Generated {len(predictions)} predictions")
    print(f"[RESULTS] CSV saved to: {predictions_file}")
    
    return predictions

def main():
    parser = argparse.ArgumentParser(description='Test AutoSign Model on Sign Language Data')
    parser.add_argument('--checkpoint', default='training_outputs/BFH/BFH_best_model.pt', 
                       help='Path to specific checkpoint file')
    parser.add_argument('--work_dir', default="./test_outputs", 
                       help='Directory to save test outputs')
    parser.add_argument('--mode', default="SI", 
                       help='Dataset mode (SI, etc.)')
    parser.add_argument('--batch_size', type=int, default=8, 
                       help='Batch size for testing')
    parser.add_argument('--additional_joints', default="1", 
                       help='Use additional joints (1 or 0)')
    parser.add_argument('--method', default="autoregressive", choices=['autoregressive', 'teacher_forcing', 'both'],
                       help='Inference method to use')
    
    args = parser.parse_args()
    args.additional_joints = True if args.additional_joints == "1" else False
    
    os.makedirs(args.work_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'[SETUP] Using device: {device}')
    
    # Load test data
    test_loader, vocab_info = setup_test_data(args)
    
    # Setup model
    from autosign.model import AutoSignLMHeadModel
    from autosign.config import AutoSignConfig
    
    config = AutoSignConfig(
        vocab_size=vocab_info['vocab_size'],
        # input_dim=86,
        # pose_embedding_length=1000,
        # max_position_embeddings=1200,
        attn_implementation='eager',
        gpt2_hf_model='aubmindlab/aragpt2-base',
        # gpt2_hf_model='/home/ubuntu/sign/Pose_To_Gloss_Arabic/aragpt2-gloss-finetuned/checkpoint-939',

    )
    
    model = AutoSignLMHeadModel(config)
    model.to(device=device)
    
    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        return
    
    checkpoint = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    checkpoint_epoch = checkpoint.get('epoch', 'unknown')
    print(f"[MODEL] Successfully loaded checkpoint from epoch {checkpoint_epoch}")
    
    # Run inference based on method choice
    if args.method == "autoregressive":
        predictions = run_inference_autoregressive(
            model, test_loader, device, vocab_info, args.work_dir, checkpoint_epoch
        )
    elif args.method == "teacher_forcing":
        predictions = run_inference_teacher_forcing(
            model, test_loader, device, vocab_info, args.work_dir, checkpoint_epoch
        )
    elif args.method == "both":
        print("Running both methods for comparison...")
        auto_preds = run_inference_autoregressive(
            model, test_loader, device, vocab_info, args.work_dir, checkpoint_epoch
        )
        tf_preds = run_inference_teacher_forcing(
            model, test_loader, device, vocab_info, args.work_dir, checkpoint_epoch
        )
        predictions = auto_preds  # Use autoregressive as default
    
    print(f"\n{'='*60}")
    print(f"Testing Complete!")
    print(f"Method: {args.method}")
    print(f"Total predictions: {len(predictions)}")
    print(f"Output directory: {args.work_dir}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()