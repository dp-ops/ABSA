import os
import json
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
import csv
from model import ASPECT_MODEL_PATH, SENTIMENT_MODEL_PATH

# Set up argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Visualize XLM-RoBERTa training metrics')
    parser.add_argument('--aspect_model_dir', type=str, default=ASPECT_MODEL_PATH,
                       help='Directory containing aspect extractor model logs')
    parser.add_argument('--sentiment_model_dir', type=str, default=SENTIMENT_MODEL_PATH,
                       help='Directory containing aspect sentiment model logs')
    parser.add_argument('--output_dir', type=str, default='metrics_plots/xlm_roberta',
                       help='Directory to save metric plots')
    return parser.parse_args()

def load_trainer_state(model_dir):
    """Load the trainer_state.json file from the model directory"""
    trainer_state_path = os.path.join(model_dir, 'checkpoints', 'trainer_state.json')
    
    # Look for trainer_state.json in subdirectories if not found
    if not os.path.exists(trainer_state_path):
        # Try to find in any checkpoint directory
        checkpoint_dirs = glob.glob(os.path.join(model_dir, 'checkpoints', 'checkpoint-*'))
        
        if checkpoint_dirs:
            # Sort by step number to get the latest
            checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]))
            trainer_state_path = os.path.join(checkpoint_dirs[-1], 'trainer_state.json')
    
    if not os.path.exists(trainer_state_path):
        print(f"No trainer_state.json found in {model_dir}")
        return None
    
    try:
        with open(trainer_state_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading trainer state: {e}")
        return None

def extract_metrics_from_log_history(log_history):
    """Extract training and evaluation metrics from log history"""
    training_steps = []
    training_loss = []
    training_epoch = []
    
    eval_steps = []
    eval_epoch = []
    eval_metrics = {}
    
    for entry in log_history:
        # Training metrics (loss)
        if 'loss' in entry and 'eval_loss' not in entry:
            step = entry.get('step', len(training_steps) + 1)
            epoch = entry.get('epoch', 0)
            training_steps.append(step)
            training_epoch.append(epoch)
            training_loss.append(entry['loss'])
            
        # Evaluation metrics
        if 'eval_loss' in entry:
            step = entry.get('step', len(eval_steps) + 1)
            epoch = entry.get('epoch', 0)
            eval_steps.append(step)
            eval_epoch.append(epoch)
            
            # Store all eval metrics
            for key, value in entry.items():
                if key.startswith('eval_') and not key == 'eval_runtime' and not key.endswith('_per_second'):
                    if key not in eval_metrics:
                        eval_metrics[key] = []
                    eval_metrics[key].append(value)
    
    return {
        'training_steps': training_steps,
        'training_epoch': training_epoch,
        'training_loss': training_loss,
        'eval_steps': eval_steps,
        'eval_epoch': eval_epoch,
        'eval_metrics': eval_metrics
    }

def save_metrics_to_csv(metrics, model_name, output_dir):
    """Save metrics to CSV file for easier analysis"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training metrics
    train_csv_path = os.path.join(output_dir, f'{model_name}_training_metrics.csv')
    with open(train_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['step', 'epoch', 'loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for i in range(len(metrics['training_steps'])):
            writer.writerow({
                'step': metrics['training_steps'][i],
                'epoch': metrics['training_epoch'][i],
                'loss': metrics['training_loss'][i]
            })
    
    print(f"Training metrics saved to {train_csv_path}")
    
    # Save evaluation metrics
    if metrics['eval_steps']:
        eval_csv_path = os.path.join(output_dir, f'{model_name}_eval_metrics.csv')
        
        # Determine all field names
        fieldnames = ['step', 'epoch']
        for metric_name in metrics['eval_metrics'].keys():
            fieldnames.append(metric_name.replace('eval_', ''))
        
        with open(eval_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for i in range(len(metrics['eval_steps'])):
                row = {
                    'step': metrics['eval_steps'][i],
                    'epoch': metrics['eval_epoch'][i] if i < len(metrics['eval_epoch']) else ''
                }
                
                for metric_name, values in metrics['eval_metrics'].items():
                    if i < len(values):
                        row[metric_name.replace('eval_', '')] = values[i]
                
                writer.writerow(row)
        
        print(f"Evaluation metrics saved to {eval_csv_path}")
    
    return train_csv_path, eval_csv_path

def plot_training_metrics(metrics, model_name, output_dir):
    """Create plots of training metrics"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training loss
    plt.figure(figsize=(12, 6))
    plt.plot(metrics['training_steps'], metrics['training_loss'])
    plt.title(f'{model_name} Training Loss (XLM-RoBERTa)')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{model_name}_training_loss.png'))
    plt.close()
    
    # Plot evaluation metrics
    for metric_name, metric_values in metrics['eval_metrics'].items():
        if len(metric_values) > 1:  # Only plot if we have more than one point
            plt.figure(figsize=(12, 6))
            plt.plot(metrics['eval_steps'], metric_values)
            plt.title(f'{model_name} {metric_name} (XLM-RoBERTa)')
            plt.xlabel('Training Steps')
            plt.ylabel(metric_name.replace('eval_', '').capitalize())
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'{model_name}_{metric_name}.png'))
            plt.close()
    
    # Plot all metrics with F1 in the name together
    f1_metrics = {k: v for k, v in metrics['eval_metrics'].items() if 'f1' in k.lower()}
    if f1_metrics:
        plt.figure(figsize=(12, 6))
        for metric_name, metric_values in f1_metrics.items():
            plt.plot(metrics['eval_steps'], metric_values, label=metric_name.replace('eval_', ''))
        plt.title(f'{model_name} F1 Scores (XLM-RoBERTa)')
        plt.xlabel('Training Steps')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{model_name}_f1_scores.png'))
        plt.close()
    
    # Plot training loss and evaluation loss together if available
    if 'eval_loss' in metrics['eval_metrics'] and metrics['training_loss']:
        # Create interpolated training loss at eval steps for comparison
        train_steps = np.array(metrics['training_steps'])
        train_loss = np.array(metrics['training_loss'])
        eval_steps = np.array(metrics['eval_steps'])
        
        # Simple interpolation for visualization
        interp_train_loss = np.interp(eval_steps, train_steps, train_loss)
        
        plt.figure(figsize=(12, 6))
        plt.plot(eval_steps, interp_train_loss, label='Training Loss')
        plt.plot(eval_steps, metrics['eval_metrics']['eval_loss'], label='Validation Loss')
        plt.title(f'{model_name} Training vs Validation Loss (XLM-RoBERTa)')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{model_name}_train_val_loss.png'))
        plt.close()

def main():
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process Aspect Extraction model metrics
    print("\nProcessing XLM-RoBERTa Aspect Extraction model metrics...")
    aspect_trainer_state = load_trainer_state(args.aspect_model_dir)
    if aspect_trainer_state and 'log_history' in aspect_trainer_state:
        aspect_metrics = extract_metrics_from_log_history(aspect_trainer_state['log_history'])
        plot_training_metrics(aspect_metrics, 'Aspect_Extraction', args.output_dir)
        # Save metrics to CSV
        csv_files = save_metrics_to_csv(aspect_metrics, 'Aspect_Extraction', args.output_dir)
        print(f"Aspect Extraction model metrics plotted to {args.output_dir} and saved as CSV")
    else:
        print("No metrics found for Aspect Extraction model")
    
    # Process Aspect Sentiment model metrics
    print("\nProcessing XLM-RoBERTa Aspect Sentiment Classification model metrics...")
    sentiment_trainer_state = load_trainer_state(args.sentiment_model_dir)
    if sentiment_trainer_state and 'log_history' in sentiment_trainer_state:
        sentiment_metrics = extract_metrics_from_log_history(sentiment_trainer_state['log_history'])
        plot_training_metrics(sentiment_metrics, 'Aspect_Sentiment', args.output_dir)
        # Save metrics to CSV
        csv_files = save_metrics_to_csv(sentiment_metrics, 'Aspect_Sentiment', args.output_dir)
        print(f"Aspect Sentiment model metrics plotted to {args.output_dir} and saved as CSV")
    else:
        print("No metrics found for Aspect Sentiment model")
    
    # Load and plot final evaluation metrics if available
    print("\nFinal evaluation metrics:")
    
    # ATE final metrics
    ate_eval_path = os.path.join(args.aspect_model_dir, 'evaluation_metrics.json')
    if os.path.exists(ate_eval_path):
        try:
            with open(ate_eval_path, 'r') as f:
                ate_metrics = json.load(f)
                print("\nXLM-RoBERTa Aspect Extraction (ATE) Final Metrics:")
                for key, value in ate_metrics.items():
                    if not key.startswith('eval_runtime') and not key.endswith('_per_second'):
                        print(f"  {key}: {value:.4f}" if isinstance(value, (int, float)) else f"  {key}: {value}")
                
                # Save final metrics to CSV
                final_csv_path = os.path.join(args.output_dir, 'ATE_final_metrics.csv')
                with open(final_csv_path, 'w', newline='') as csvfile:
                    fieldnames = ['metric', 'value']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for key, value in ate_metrics.items():
                        if not key.startswith('eval_runtime') and not key.endswith('_per_second'):
                            writer.writerow({'metric': key, 'value': value})
                print(f"Final ATE metrics saved to {final_csv_path}")
        except Exception as e:
            print(f"Error loading ATE evaluation metrics: {e}")
    
    # ASC final metrics
    asc_eval_path = os.path.join(args.sentiment_model_dir, 'evaluation_metrics.json')
    if os.path.exists(asc_eval_path):
        try:
            with open(asc_eval_path, 'r') as f:
                asc_metrics = json.load(f)
                print("\nXLM-RoBERTa Aspect Sentiment Classification (ASC) Final Metrics:")
                for key, value in asc_metrics.items():
                    if not key.startswith('eval_runtime') and not key.endswith('_per_second'):
                        print(f"  {key}: {value:.4f}" if isinstance(value, (int, float)) else f"  {key}: {value}")
                
                # Save final metrics to CSV
                final_csv_path = os.path.join(args.output_dir, 'ASC_final_metrics.csv')
                with open(final_csv_path, 'w', newline='') as csvfile:
                    fieldnames = ['metric', 'value']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for key, value in asc_metrics.items():
                        if not key.startswith('eval_runtime') and not key.endswith('_per_second'):
                            writer.writerow({'metric': key, 'value': value})
                print(f"Final ASC metrics saved to {final_csv_path}")
        except Exception as e:
            print(f"Error loading ASC evaluation metrics: {e}")
    
    print(f"\nAll XLM-RoBERTa model plots saved to {args.output_dir}")
    print(f"All XLM-RoBERTa model metrics saved as CSV files in {args.output_dir}")

if __name__ == "__main__":
    main() 