import os
import json
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set up argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Visualize training metrics')
    parser.add_argument('--aspect_model_dir', type=str, default='saved_models/aspect_extractor_model',
                       help='Directory containing aspect extractor model logs')
    parser.add_argument('--sentiment_model_dir', type=str, default='saved_models/aspect_sentiment_model',
                       help='Directory containing aspect sentiment model logs')
    parser.add_argument('--output_dir', type=str, default='metrics_plots',
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
    
    eval_steps = []
    eval_metrics = {}
    
    for entry in log_history:
        # Training metrics (loss)
        if 'loss' in entry and 'eval_loss' not in entry:
            step = entry.get('step', len(training_steps) + 1)
            training_steps.append(step)
            training_loss.append(entry['loss'])
            
        # Evaluation metrics
        if 'eval_loss' in entry:
            step = entry.get('step', len(eval_steps) + 1)
            eval_steps.append(step)
            
            # Store all eval metrics
            for key, value in entry.items():
                if key.startswith('eval_') and not key == 'eval_runtime' and not key.endswith('_per_second'):
                    if key not in eval_metrics:
                        eval_metrics[key] = []
                    eval_metrics[key].append(value)
    
    return {
        'training_steps': training_steps,
        'training_loss': training_loss,
        'eval_steps': eval_steps,
        'eval_metrics': eval_metrics
    }

def plot_training_metrics(metrics, model_name, output_dir):
    """Create plots of training metrics"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['training_steps'], metrics['training_loss'])
    plt.title(f'{model_name} Training Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{model_name}_training_loss.png'))
    plt.close()
    
    # Plot evaluation metrics
    for metric_name, metric_values in metrics['eval_metrics'].items():
        if len(metric_values) > 1:  # Only plot if we have more than one point
            plt.figure(figsize=(10, 6))
            plt.plot(metrics['eval_steps'], metric_values)
            plt.title(f'{model_name} {metric_name}')
            plt.xlabel('Training Steps')
            plt.ylabel(metric_name.replace('eval_', '').capitalize())
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'{model_name}_{metric_name}.png'))
            plt.close()
    
    # Plot all metrics with F1 in the name together
    f1_metrics = {k: v for k, v in metrics['eval_metrics'].items() if 'f1' in k.lower()}
    if f1_metrics:
        plt.figure(figsize=(10, 6))
        for metric_name, metric_values in f1_metrics.items():
            plt.plot(metrics['eval_steps'], metric_values, label=metric_name.replace('eval_', ''))
        plt.title(f'{model_name} F1 Scores')
        plt.xlabel('Training Steps')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{model_name}_f1_scores.png'))
        plt.close()

def main():
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process Aspect Extraction model metrics
    print("Processing Aspect Extraction model metrics...")
    aspect_trainer_state = load_trainer_state(args.aspect_model_dir)
    if aspect_trainer_state and 'log_history' in aspect_trainer_state:
        aspect_metrics = extract_metrics_from_log_history(aspect_trainer_state['log_history'])
        plot_training_metrics(aspect_metrics, 'Aspect_Extraction', args.output_dir)
        print(f"Aspect Extraction model metrics plotted to {args.output_dir}")
    else:
        print("No metrics found for Aspect Extraction model")
    
    # Process Aspect Sentiment model metrics
    print("Processing Aspect Sentiment Classification model metrics...")
    sentiment_trainer_state = load_trainer_state(args.sentiment_model_dir)
    if sentiment_trainer_state and 'log_history' in sentiment_trainer_state:
        sentiment_metrics = extract_metrics_from_log_history(sentiment_trainer_state['log_history'])
        plot_training_metrics(sentiment_metrics, 'Aspect_Sentiment', args.output_dir)
        print(f"Aspect Sentiment model metrics plotted to {args.output_dir}")
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
                print("\nAspect Extraction (ATE) Final Metrics:")
                for key, value in ate_metrics.items():
                    if not key.startswith('eval_runtime') and not key.endswith('_per_second'):
                        print(f"  {key}: {value:.4f}" if isinstance(value, (int, float)) else f"  {key}: {value}")
        except Exception as e:
            print(f"Error loading ATE evaluation metrics: {e}")
    
    # ASC final metrics
    asc_eval_path = os.path.join(args.sentiment_model_dir, 'evaluation_metrics.json')
    if os.path.exists(asc_eval_path):
        try:
            with open(asc_eval_path, 'r') as f:
                asc_metrics = json.load(f)
                print("\nAspect Sentiment Classification (ASC) Final Metrics:")
                for key, value in asc_metrics.items():
                    if not key.startswith('eval_runtime') and not key.endswith('_per_second'):
                        print(f"  {key}: {value:.4f}" if isinstance(value, (int, float)) else f"  {key}: {value}")
        except Exception as e:
            print(f"Error loading ASC evaluation metrics: {e}")
    
    print(f"\nAll plots saved to {args.output_dir}")

if __name__ == "__main__":
    main() 