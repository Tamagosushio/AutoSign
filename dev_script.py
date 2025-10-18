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
from utils.metrics import wer_list
import argparse
import zipfile

def custom_collate_fn(batch):
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

def setup_dev_data(args):
    train_csv = f"annotations_v2/{args.mode}/train.txt"
    dev_csv = f"annotations_v2/{args.mode}/dev.txt"
    
    train_processed, dev_processed, vocab_map, inv_vocab_map, vocab_list = load_and_process_text_data(
        train_csv, dev_csv, target_column='gloss'
    )
    
    dataset_dev = PoseDatasetV2(
        dataset_name2="isharah",
        label_csv=dev_csv,
        split_type="dev",
        target_enc_df=dev_processed,
        augmentations=False,
        pose_data_path=f"data/pose_data_isharah1000_hands_lips_body_May12.pkl"
    )
    
    dev_loader = DataLoader(
        dataset_dev, 
        batch_size=4 if args.method == 'autoregressive' else args.batch_size,
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
    
    return dev_loader, vocab_info

def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint

def remove_duplicates(text):
    if not text or len(text.split()) < 2:
        return text
    
    words = text.split()
    result = [words[0]]
    
    for word in words[1:]:
        if word != result[-1]:
            result.append(word)
    
    return ' '.join(result)

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

def run_dev_evaluation_teacher_forcing(model, dev_loader, device, vocab_info, work_dir, checkpoint_epoch):
    model.eval()
    predictions = []
    ground_truths = []
    
    submission_file = f"{work_dir}/dev.csv"
    
    with open(submission_file, "w", encoding='utf-8') as sub_file:
        sub_file.write("id,gloss\n")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm.tqdm(dev_loader, desc="Teacher Forcing Eval", ncols=100)):
                inputs = {k: v for k, v in batch.items() if k != 'file_path'}
                inputs = send_inputs_to_device(inputs, device)
                
                outputs = model(**inputs)
                pose_seq_len = inputs['pose_values'].size(1)
                text_logits = outputs.logits[:, pose_seq_len:, :]
                batch_predictions = decode_predictions(text_logits, vocab_info)
                
                from utils.utils import decode_labels
                batch_ground_truths = decode_labels(inputs['labels'], vocab_info)
                
                file_names = batch['file_path']
                for i, (file_name, prediction, ground_truth) in enumerate(zip(file_names, batch_predictions, batch_ground_truths)):
                    clean_prediction = remove_duplicates(prediction.strip())
                    
                    predictions.append(clean_prediction)
                    ground_truths.append(ground_truth)
                    
                    sub_file.write(f"{file_name},{clean_prediction}\n")
    
    correct = sum(1 for pred, gt in zip(predictions, ground_truths) 
                  if pred.strip() == gt.strip())
    accuracy = correct / len(predictions) if predictions else 0.0
    
    try:
        wer_results = wer_list(predictions, ground_truths)
        wer_score = wer_results["wer"]
    except Exception:
        wer_score = -1
    
    zip_file = f"{work_dir}/dev.zip"
    with zipfile.ZipFile(zip_file, 'w') as zipf:
        zipf.write(submission_file, 'dev.csv')
    
    return predictions, ground_truths, accuracy, wer_score

def run_dev_evaluation_autoregressive(model, dev_loader, device, vocab_info, work_dir, checkpoint_epoch):
    model.eval()
    predictions = []
    ground_truths = []
    
    submission_file = f"{work_dir}/dev.csv"
    
    with open(submission_file, "w", encoding='utf-8') as sub_file:
        sub_file.write("id,gloss\n")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm.tqdm(dev_loader, desc="Autoregressive Eval", ncols=100)):
                inputs = {k: v for k, v in batch.items() if k != 'file_path'}
                inputs = send_inputs_to_device(inputs, device)
                
                pose_values = inputs['pose_values']
                file_names = batch['file_path']
                
                from utils.utils import decode_labels
                batch_ground_truths = decode_labels(inputs['labels'], vocab_info)
                
                for i in range(pose_values.size(0)):
                    single_pose = pose_values[i:i+1]
                    
                    generated_ids = generate_autoregressive(
                        model, single_pose, vocab_info, device, 
                        max_length=12, temperature=0.7, repetition_penalty=1.2
                    )
                    
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
                    
                    predictions.append(clean_prediction)
                    ground_truths.append(ground_truth)
                    
                    sub_file.write(f"{file_names[i]},{clean_prediction}\n")
    
    correct = sum(1 for pred, gt in zip(predictions, ground_truths) 
                  if pred.strip() == gt.strip())
    accuracy = correct / len(predictions) if predictions else 0.0
    
    try:
        wer_results = wer_list(predictions, ground_truths)
        wer_score = wer_results["wer"]
    except Exception:
        wer_score = -1
    
    zip_file = f"{work_dir}/dev.zip"
    with zipfile.ZipFile(zip_file, 'w') as zipf:
        zipf.write(submission_file, 'dev.csv')
    
    return predictions, ground_truths, accuracy, wer_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='./training_outputs_new/checkpoint_epoch_8.pt')
    parser.add_argument('--work_dir', default="./dev_evaluation")
    parser.add_argument('--mode', default="SI")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--method', default='autoregressive', choices=['teacher_forcing', 'autoregressive', 'both'])
    
    args = parser.parse_args()
    
    os.makedirs(args.work_dir, exist_ok=True)
    
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    
    dev_loader, vocab_info = setup_dev_data(args)
    
    from autosign.model import AutoSignLMHeadModel
    from autosign.config import AutoSignConfig
    
    config = AutoSignConfig(
        vocab_size=vocab_info['vocab_size'],
        # input_dim=86,
        pose_embedding_length=1000,
        max_position_embeddings=1200,
        attn_implementation='eager',
        gpt2_hf_model='aubmindlab/aragpt2-base',
    )
    
    model = AutoSignLMHeadModel(config)
    model.to(device=device)
    
    checkpoint = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    checkpoint_epoch = checkpoint.get('epoch', 'unknown')
    
    if args.method == 'teacher_forcing':
        predictions, ground_truths, accuracy, wer_score = run_dev_evaluation_teacher_forcing(
            model, dev_loader, device, vocab_info, args.work_dir, checkpoint_epoch
        )
        print(f"Teacher Forcing - Accuracy: {accuracy:.4f}, WER: {wer_score:.4f}")
        
    elif args.method == 'autoregressive':
        predictions, ground_truths, accuracy, wer_score = run_dev_evaluation_autoregressive(
            model, dev_loader, device, vocab_info, args.work_dir, checkpoint_epoch
        )
        print(f"Autoregressive - Accuracy: {accuracy:.4f}, WER: {wer_score:.4f}")
        
    elif args.method == 'both':
        tf_preds, tf_gts, tf_acc, tf_wer = run_dev_evaluation_teacher_forcing(
            model, dev_loader, device, vocab_info, f"{args.work_dir}_tf", checkpoint_epoch
        )
        
        ar_preds, ar_gts, ar_acc, ar_wer = run_dev_evaluation_autoregressive(
            model, dev_loader, device, vocab_info, f"{args.work_dir}_ar", checkpoint_epoch
        )
        
        print(f"Teacher Forcing - Accuracy: {tf_acc:.4f}, WER: {tf_wer:.4f}")
        print(f"Autoregressive - Accuracy: {ar_acc:.4f}, WER: {ar_wer:.4f}")
        print(f"Improvement: {ar_acc - tf_acc:+.4f} accuracy, {tf_wer - ar_wer:+.4f} WER")

if __name__ == '__main__':
    main()