import os
import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.datasetv2 import PoseDatasetV2
from collections import Counter
from typing import List, Dict, Tuple
import numpy as np
import time
from utils.utils import GaussianNoise, send_inputs_to_device, decode_predictions, decode_labels, load_and_process_text_data
from utils.metrics import wer_list
import random
import argparse
from torch.nn.utils.rnn import pad_sequence
from utils.utils import generate_autoregressive
from utils.metrics import wer_list

torch.set_float32_matmul_precision('high')

def custom_collate_fn(batch):
    """Custom collate function to handle variable-length sequences"""
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

def setup_training_data(mode):
    """Setup data loaders"""
    train_csv = f"annotations_v2/{mode}/train.txt" 
    dev_csv = f"annotations_v2/{mode}/dev.txt"   

    train_processed, dev_processed, vocab_map, inv_vocab_map, vocab_list = load_and_process_text_data(
        train_csv, dev_csv, target_column='gloss'
    )
    
    dataset_train = PoseDatasetV2(
        dataset_name2="isharah",
        label_csv=train_csv,
        split_type="train",
        target_enc_df=train_processed,
        augmentations=True,
        augmentation_config='aggressive',
        transform=transforms.Compose([GaussianNoise()])
    )
    
    dataset_dev = PoseDatasetV2(
        dataset_name2="isharah", 
        label_csv=dev_csv,
        split_type="dev",
        target_enc_df=dev_processed,
        augmentations=False
    )
    
    train_loader = DataLoader(
        dataset_train, 
        batch_size=4, 
        shuffle=True, 
        num_workers=10,
        collate_fn=custom_collate_fn 
    )
    
    dev_loader = DataLoader(
        dataset_dev, 
        batch_size=4, 
        shuffle=False, 
        num_workers=8,
        collate_fn=custom_collate_fn  
    )
    
    vocab_info = {
        'vocab_map': vocab_map, 
        'inv_vocab_map': inv_vocab_map, 
        'vocab_list': vocab_list,
        'vocab_size': len(vocab_list)
    }
    
    print(f"[DATA] Training: {len(dataset_train)}, Dev: {len(dataset_dev)}, Vocab: {len(vocab_list)}")
    return train_loader, dev_loader, vocab_info

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, val_wer, checkpoint_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_wer': val_wer,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def save_best_model(model, optimizer, epoch, train_loss, val_loss, val_wer, work_dir):
    """Save the best model based on WER"""
    best_model_path = os.path.join(work_dir, 'best_model.pt')
    save_checkpoint(model, optimizer, epoch, train_loss, val_loss, val_wer, best_model_path)
    print(f"NEW BEST MODEL SAVED! WER: {val_wer:.4f}")

import torch
import torch.nn.functional as F
import tqdm
import os
from utils.utils import decode_labels

def apply_repetition_penalty(logits, input_ids, repetition_penalty, vocab_info):
    """Apply repetition penalty to logits"""
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
                           max_length=20, temperature=0.8, repetition_penalty=1.2):
    """Generate text autoregressively with repetition penalty"""
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
            # next_token_logits = outputs.logits[:, pose_seq_len + current_text_len - 1, :].clone()
            next_token_logits = outputs.logits[:, -1, :].clone()

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

def evaluate_model_with_wer_autoregressive(model, dataloader, device, vocab_info, work_dir, epoch):
    """
    Enhanced evaluation with autoregressive generation and WER metrics
    
    Args:
        model: The AutoSign model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        vocab_info: Vocabulary information dictionary
        work_dir: Directory to save prediction outputs
        epoch: Current epoch number
        
    Returns:
        Tuple: Average loss and WER score
    """
    print(f"Starting autoregressive evaluation with WER for epoch {epoch + 1}...")
    
    model.eval()
    all_predictions = []
    all_ground_truths = []
    
    os.makedirs(f"{work_dir}/pred_outputs", exist_ok=True)
    predictions_file = f"{work_dir}/pred_outputs/predictions_autoregressive_epoch_{epoch+1}.txt"
    
    with open(predictions_file, "w", encoding='utf-8') as pred_file:
        pred_file.write(f"Epoch {epoch+1} Autoregressive Predictions\n")
        pred_file.write("=" * 60 + "\n\n")
        
        with torch.no_grad():
            sample_count = 0
            
            for batch_idx, batch in enumerate(tqdm.tqdm(dataloader, desc="Autoregressive Eval", ncols=100)):
                
                inputs = {k: v for k, v in batch.items() if k != 'file_path'}
                inputs = send_inputs_to_device(inputs, device)
                
                pose_values = inputs['pose_values']
                file_names = batch['file_path']
                
                # Get ground truth from labels
                batch_ground_truths = decode_labels(inputs['labels'], vocab_info)
                
                for i in range(pose_values.size(0)):
                    single_pose = pose_values[i:i+1]
                    
                    # Generate using autoregressive method
                    generated_ids = generate_autoregressive(
                        model, single_pose, vocab_info, device, 
                        max_length=12, temperature=0.9, repetition_penalty=1.0
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
                    ground_truth = batch_ground_truths[i]
                    
                    all_predictions.append(clean_prediction)
                    all_ground_truths.append(ground_truth)
                    
                    sample_count += 1
                    
                    # Write to prediction file
                    pred_file.write(f"GT:   {ground_truth}\n")
                    pred_file.write(f"Pred: {clean_prediction}\n")
                    
                    match = "yes" if clean_prediction.strip() == ground_truth.strip() else "no"
                    pred_file.write(f"Match: {match}\n")
                    pred_file.write("-" * 40 + "\n\n")
    
    wer_results = wer_list(all_predictions, all_ground_truths)
    wer_score = wer_results["wer"]
    
    # Calculate accuracy 
    correct = sum(1 for pred, gt in zip(all_predictions, all_ground_truths) 
                  if pred.strip() == gt.strip())
    accuracy = correct / len(all_predictions) if all_predictions else 0.0
    

    avg_loss = 0.0 
    
    model.train()
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Autoregressive evaluation complete:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  WER: {wer_score:.4f}")
    print(f"  Predictions saved to: {predictions_file}")
    
    return avg_loss, wer_score


def enhanced_training_pipeline_with_wer_and_scheduler(mode):
    """Training pipeline with optional learning rate scheduling"""
    print("="*60)
    print("="*60)
    
    # Setup (same as before)
    train_loader, dev_loader, vocab_info = setup_training_data(mode)
    work_dir = f"./training_outputs/12"
    os.makedirs(work_dir, exist_ok=True)
    
    # Model setup
    from autosign.model import AutoSignLMHeadModel
    from autosign.config import AutoSignConfig
    
    config = AutoSignConfig(
        vocab_size=vocab_info['vocab_size'],
        attn_implementation='eager',
        gpt2_hf_model='aubmindlab/aragpt2-base',
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoSignLMHeadModel(config)
    model.to(device=device)
    
    EPOCHS = 100
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-4, 
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    scheduler = None
    if config.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,      
            T_mult=2,  
            eta_min=1e-6 
        )
        
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode='min', factor=0.5, patience=5
        # )
    
    scaler = torch.amp.GradScaler('cuda', enabled=True)    
    
    PATIENCE = 10
    best_wer = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    train_losses = []
    val_losses, val_wer_scores = [], []
    learning_rates = [] 
    
    print(f"Early stopping patience: {PATIENCE} epochs")
    print(f"Use scheduler: {config.use_scheduler}")
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*50}")
        print(f"[EPOCH {epoch + 1}/{EPOCHS}] Starting...")
        print(f"[EPOCH {epoch + 1}] Current LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        model.train()
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        epoch_losses = []
        
        progress_bar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch + 1}')
        for batch_idx, batch in enumerate(progress_bar):
            
            inputs = {k: v for k, v in batch.items() if k != 'file_path'}
            inputs = send_inputs_to_device(inputs, device)
            
            optimizer.zero_grad()
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                outputs = model(**inputs)
            
            scaler.scale(outputs.loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_losses.append(outputs.loss.item())
            
            if batch_idx % 50 == 0:
                progress_bar.set_postfix({
                    'Loss': f'{outputs.loss.item():.4f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.2e}',
                    'Best WER': f'{best_wer:.4f}',
                    'Patience': f'{patience_counter}/{PATIENCE}',
                    'Scheduler': 'On' if scheduler else 'Off'
                })
        
        # Calculate training loss average
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        
        print(f"[EPOCH {epoch + 1}] Evaluating with WER...")
        val_loss, val_wer = evaluate_model_with_wer_autoregressive(
            model, dev_loader, device, vocab_info, work_dir, epoch
        )
        
        # scheduler step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_wer) 
            else:
                scheduler.step() 
            print(f"[EPOCH {epoch + 1}] Scheduler stepped (LR may have changed)")
        else:
            print(f"[EPOCH {epoch + 1}] No scheduler - LR remains constant")
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        val_wer_scores.append(val_wer)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        # Print results
        print(f"[EPOCH {epoch + 1}] RESULTS:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val WER: {val_wer:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Best WER: {best_wer:.4f} (Epoch {best_epoch + 1})")
        
        if (epoch + 1) % 10 == 0:
            regular_checkpoint_path = os.path.join(work_dir, f'checkpoint_epoch_{epoch + 1}.pt')
            # save_checkpoint(model, optimizer, epoch + 1, avg_train_loss, val_loss, val_wer, regular_checkpoint_path)# I commented it to save space
        
        if val_wer < best_wer:
            # New best model!
            best_wer = val_wer
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            save_best_model(model, optimizer, epoch + 1, avg_train_loss, val_loss, val_wer, work_dir)
            
        else:
            # No improvement
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{PATIENCE}")
            
            # Early stopping check
            if patience_counter >= PATIENCE:
                print(f"\nEARLY STOPPING TRIGGERED!")
                print(f"No improvement for {PATIENCE} epochs.")
                print(f"Best WER: {best_wer:.4f} at epoch {best_epoch + 1}")
                break
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE!")
    print(f"Best WER: {best_wer:.4f} (Epoch {best_epoch + 1})")
    print(f"Total epochs trained: {len(train_losses)}")
    print(f"Final LR: {optimizer.param_groups[0]['lr']:.2e}")
    print(f"Best model saved at: {os.path.join(work_dir, 'best_model.pt')}")
    print(f"{'='*60}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_wer_scores': val_wer_scores,
        'learning_rates': learning_rates, 
        'best_wer': best_wer,
        'best_epoch': best_epoch,
        'model': model,
        'vocab_info': vocab_info,
        'scheduler_used': scheduler is not None
    }


def plot_training_curves(results):
    """Plot training curves including learning rate"""
    try:
        import matplotlib.pyplot as plt
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        epochs = range(1, len(results['train_losses']) + 1)
        
        # Loss plot
        ax1.plot(epochs, results['train_losses'], label='Train Loss', color='blue', linewidth=2)
        ax1.plot(epochs, results['val_losses'], label='Val Loss', color='orange', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # WER plot
        ax2.plot(epochs, results['val_wer_scores'], label='Val WER', color='red', linewidth=2, marker='o', markersize=3)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('WER')
        ax2.set_title('Validation WER (Lower is Better)')
        ax2.grid(True, alpha=0.3)
        
        ax3.plot(epochs, results['learning_rates'], label='Learning Rate', color='green', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_yscale('log')  # Log scale for LR
        ax3.grid(True, alpha=0.3)
        
        best_epoch = results['best_epoch']
        for ax in [ax1, ax2, ax3]:
            ax.axvline(x=best_epoch + 1, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        print(f"Training curves with scheduler saved!")
        
    except ImportError:
        print("matplotlib not available, skipping plots")


def plot_training_curves_with_wer(results):
    """Plot training curves with WER
    
    Args:
        results (Dict): Training results containing losses and WER scores
        Plots training and validation losses, and validation WER scores.
        
        Returns:
        None: Saves the plots to files.
        """
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        epochs = range(1, len(results['train_losses']) + 1)
        
        ax1.plot(epochs, results['train_losses'], label='Train Loss', color='blue', linewidth=2)
        ax1.plot(epochs, results['val_losses'], label='Val Loss', color='orange', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(1, len(epochs))
        
        best_epoch = results['best_epoch']
        best_val_loss = results['val_losses'][best_epoch]
        ax1.axvline(x=best_epoch + 1, color='red', linestyle='--', alpha=0.7, label=f'Best Model (Epoch {best_epoch + 1})')
        ax1.scatter([best_epoch + 1], [best_val_loss], color='red', s=100, zorder=5)
        ax1.legend(fontsize=11)
        
        # WER plot
        ax2.plot(epochs, results['val_wer_scores'], label='Val WER', color='red', linewidth=2, marker='o', markersize=4)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('WER', fontsize=12)
        ax2.set_title('Validation WER (Lower is Better)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(1, len(epochs))
        
        best_wer = results['best_wer']
        ax2.axvline(x=best_epoch + 1, color='green', linestyle='--', alpha=0.7, label=f'Best WER: {best_wer:.4f}')
        ax2.scatter([best_epoch + 1], [best_wer], color='green', s=100, zorder=5)
        ax2.legend(fontsize=11)
        
        ax2.text(0.02, 0.98, f'Best WER: {best_wer:.4f}\nBest Epoch: {best_epoch + 1}', 
                transform=ax2.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = 'training_curves_with_early_stopping.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"\nTraining curves saved to: {plot_filename}")
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, results['train_losses'], label='Train Loss', color='blue', linewidth=2)
        plt.plot(epochs, results['val_losses'], label='Val Loss', color='orange', linewidth=2)
        plt.axvline(x=best_epoch + 1, color='red', linestyle='--', alpha=0.7, label=f'Best Model (Epoch {best_epoch + 1})')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('loss_curves.png', dpi=300, bbox_inches='tight')
        
        # WER only
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, results['val_wer_scores'], label='Val WER', color='red', linewidth=2, marker='o', markersize=4)
        plt.axvline(x=best_epoch + 1, color='green', linestyle='--', alpha=0.7, label=f'Best WER: {best_wer:.4f}')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('WER', fontsize=12)
        plt.title('Validation WER (Lower is Better)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('wer_curve.png', dpi=300, bbox_inches='tight')
        
        
    except ImportError:
        print("matplotlib not available, skipping plots")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='SI', 
                        choices=['US', 'SI'],
                        help='Training mode: US or SI')
    args = parser.parse_args()

    print(f"Starting training with mode: {args.mode}")
    # results = enhanced_training_pipeline_with_wer(args.mode)
    results = enhanced_training_pipeline_with_wer_and_scheduler(args.mode)
    plot_training_curves_with_wer(results)
    print("[DONE] Enhanced training with early stopping completed!")






