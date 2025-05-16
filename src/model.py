#TODO F1OnPlateauCallback: 
# - Make it so that if it reaches the plateau of 1e-6, and not improve after 30 epochs, it stops training

import json
import time
import os
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
import re
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForTokenClassification
from transformers import pipeline
from transformers import TrainerCallback
from datasets import Dataset, DatasetDict
from seqeval.metrics import classification_report
from sklearn.metrics import classification_report as cls_report

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def align_tokens_and_labels(original_tokens, original_labels, tokenizer):
    """
    Align original tokens and their BIO labels with BERT tokenizer output.
    This handles the subword tokenization that BERT does.
    
    Args:
        original_tokens: List of original tokens from the dataset
        original_labels: List of BIO labels corresponding to original tokens
        tokenizer: The BERT tokenizer
        
    Returns:
        List of aligned labels for BERT tokenized input (including special tokens)
    """
    # Convert to lowercase if needed (depending on if the tokenizer is cased or uncased)
    bert_tokens = []
    bert_labels = []
    
    # Add [CLS] token
    bert_tokens.append("[CLS]")
    bert_labels.append("O")  # [CLS] token is always outside
    
    # Process each original token and align with BERT tokens
    for orig_token, orig_label in zip(original_tokens, original_labels):
        # Tokenize the original token to get subwords
        subwords = tokenizer.tokenize(orig_token)
        
        # If no subwords were produced (rare, but could happen with some tokens)
        if not subwords:
            continue
            
        # Add the subwords and their labels
        for i, subword in enumerate(subwords):
            bert_tokens.append(subword)
            
            # First subword gets the original label
            if i == 0:
                bert_labels.append(orig_label)
            else:
                # For subsequent subwords:
                # - If original was B-ASP, subsequent is I-ASP
                # - If original was I-ASP, subsequent remains I-ASP
                # - If original was O, subsequent remains O
                if orig_label == "B-ASP":
                    bert_labels.append("I-ASP")
                else:
                    bert_labels.append(orig_label)
    
    # Add [SEP] token
    bert_tokens.append("[SEP]")
    bert_labels.append("O")  # [SEP] token is always outside
    
    # Verify the alignment
    if len(bert_tokens) != len(bert_labels):
        logger.warning(f"Mismatch in aligned tokens ({len(bert_tokens)}) and labels ({len(bert_labels)})")
    
    return bert_tokens, bert_labels

def convert_aligned_labels_to_ids(aligned_labels, label_map):
    """Convert string labels to IDs using the label map"""
    return [label_map.get(label, 0) for label in aligned_labels]

# Constants
MODEL_NAME = "nlpaueb/bert-base-greek-uncased-v1"
SAVED_MODELS_DIR = "saved_models"
NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_LENGTH = 128  # Maximum sequence length

# Create saved_models directory if it doesn't exist
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# Aspect term extraction label mapping
ASPECT_LABEL_MAP = {"O": 0, "B-ASP": 1, "I-ASP": 2}
ASPECT_NUM_LABELS = len(ASPECT_LABEL_MAP)
ASPECT_LABEL_MAP_INVERSE = {v: k for k, v in ASPECT_LABEL_MAP.items()}

# Aspect sentiment classification label mapping
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
SENTIMENT_NUM_LABELS = len(SENTIMENT_LABELS)

# Model paths
ASPECT_MODEL_PATH = f"{SAVED_MODELS_DIR}/aspect_extractor_model"
SENTIMENT_MODEL_PATH = f"{SAVED_MODELS_DIR}/aspect_sentiment_model"

# ================ DATASET LOADING FUNCTIONS ================

def load_aspect_dataset(file_path, tokenizer, label_map=ASPECT_LABEL_MAP):
    """
    Load and preprocess the ATE (Aspect Term Extraction) dataset
    """
    logger.info(f"Loading aspect extraction dataset from {file_path}")
    
    # Load the JSON Lines file (one JSON object per line)
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))

    # Convert to a format compatible with the datasets library
    formatted_data = []
    for item in data:
        # Get the text
        text = item['text']
        
        # Check if we have pre-generated tokens and BIO labels
        if 'tokens' in item and 'bio_labels' in item:
            orig_tokens = item['tokens']
            orig_bio_labels = item['bio_labels']
            
            # Align the original tokens and labels with BERT tokenization
            aligned_tokens, aligned_labels = align_tokens_and_labels(orig_tokens, orig_bio_labels, tokenizer)
            
            # Convert aligned labels to IDs
            label_ids = convert_aligned_labels_to_ids(aligned_labels, label_map)
            
            # Tokenize the text with aligned tokens
            encoding = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors=None  # Return Python lists
            )
            
            # Make sure label_ids matches the length of input_ids (pad if needed)
            if len(label_ids) < len(encoding['input_ids']):
                # Pad with O (0)
                label_ids = label_ids + [0] * (len(encoding['input_ids']) - len(label_ids))
            elif len(label_ids) > len(encoding['input_ids']):
                # Truncate
                label_ids = label_ids[:len(encoding['input_ids'])]
                
            # Create entry with consistent keys
            formatted_data.append({
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'token_type_ids': encoding['token_type_ids'],
                'labels': label_ids
            })
        else:
            # For data without pre-tokenized text and labels, just use text
            # This is a fallback, but should be rare
            encoding = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors=None  # Return Python lists
            )
            
            # Initialize all tokens with 'O' label (0)
            labels = [0] * len(encoding['input_ids'])
            
            formatted_data.append({
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'token_type_ids': encoding['token_type_ids'],
                'labels': labels
            })

    return Dataset.from_list(formatted_data)

def load_sentiment_dataset(file_path, tokenizer):
    """
    Load and preprocess the ASC (Aspect Sentiment Classification) dataset
    """
    logger.info(f"Loading sentiment classification dataset from {file_path}")
    
    # Load the JSON Lines file (one JSON object per line)
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    
    formatted_data = []
    for item in data:
        text = item['text']
        # Create entries for each aspect and its sentiment
        for aspect_info in item['aspects']:
            aspect = aspect_info['aspect']
            # Check if sentiment_id exists, otherwise try to get it from 'sentiment'
            if 'sentiment_id' in aspect_info:
                sentiment_id = aspect_info['sentiment_id']
            elif 'sentiment' in aspect_info:
                # Map sentiment text to id if necessary
                sentiment_text = aspect_info['sentiment'].lower()
                if sentiment_text == 'negative':
                    sentiment_id = 0
                elif sentiment_text == 'neutral':
                    sentiment_id = 1
                elif sentiment_text == 'positive':
                    sentiment_id = 2
                else:
                    logger.warning(f"Unknown sentiment value: {sentiment_text}, defaulting to neutral")
                    sentiment_id = 1
            else:
                logger.warning("No sentiment information found in aspect, defaulting to neutral")
                sentiment_id = 1
            
            # Encode the text and aspect as a pair
            encoding = tokenizer(
                text, 
                aspect, 
                padding="max_length", 
                truncation=True, 
                max_length=MAX_LENGTH,
                return_tensors=None  # Return Python lists
            )
            
            formatted_data.append({
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'token_type_ids': encoding['token_type_ids'],
                'labels': sentiment_id
            })
    
    return Dataset.from_list(formatted_data)

# ================ METRIC COMPUTATION FUNCTIONS ================

def compute_ate_metrics(p):
    """
    Compute metrics for Aspect Term Extraction
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [ASPECT_LABEL_MAP_INVERSE[p_val] for (p_val, l_val) in zip(prediction, label) if l_val != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [ASPECT_LABEL_MAP_INVERSE[l_val] for (p_val, l_val) in zip(prediction, label) if l_val != -100]
        for prediction, label in zip(predictions, labels)
    ]

    try:
        # Set zero_division=1 to avoid warnings and errors
        results_dict = classification_report(true_labels, true_predictions, output_dict=True, zero_division=1)
        
        # Initialize metrics to return 
        metrics_to_return = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

        # Safely extract macro average scores
        if "macro avg" in results_dict:
            macro_avg_stats = results_dict["macro avg"]
            precision_val = macro_avg_stats.get("precision", 0.0)
            recall_val = macro_avg_stats.get("recall", 0.0)
            f1_val = macro_avg_stats.get("f1-score", 0.0)

            metrics_to_return["precision"] = 0.0 if np.isnan(precision_val) else precision_val
            metrics_to_return["recall"] = 0.0 if np.isnan(recall_val) else recall_val
            metrics_to_return["f1"] = 0.0 if np.isnan(f1_val) else f1_val
            
            if np.isnan(macro_avg_stats.get("f1-score", 0.0)) and metrics_to_return["f1"] == 0.0:
                 logger.warning(
                    "ATE macro F1 score was NaN (likely no aspects predicted/found), reported as 0.0."
                 )
        else:
            logger.warning("ATE metrics: 'macro avg' not found in classification_report output.")
            
        return metrics_to_return

    except Exception as e:
        logger.warning(f"Error computing ATE metrics: {e}. True labels sample: {str(true_labels[:1]) if true_labels else '[]'}, Preds sample: {str(true_predictions[:1]) if true_predictions else '[]'}")
        # Return default metrics on error
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

def compute_asc_metrics(p):
    """
    Compute metrics for Aspect Sentiment Classification
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    
    try:
        # Set zero_division=1 to avoid warnings and errors
        results = cls_report(
            labels, 
            predictions, 
            target_names=SENTIMENT_LABELS, 
            output_dict=True, 
            digits=4,
            zero_division=1
        )
        
        return {
            "accuracy": results["accuracy"],
            "macro_precision": results["macro avg"]["precision"],
            "macro_recall": results["macro avg"]["recall"],
            "macro_f1": results["macro avg"]["f1-score"],
            "neg_f1": results["negative"]["f1-score"],
            "neu_f1": results["neutral"]["f1-score"],
            "pos_f1": results["positive"]["f1-score"]
        }
    except Exception as e:
        logger.warning(f"Error computing ASC metrics: {e}")
        # Return default metrics on error
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "neg_f1": 0.0,
            "neu_f1": 0.0,
            "pos_f1": 0.0
        }

# ================ MODEL TRAINING FUNCTIONS ================

def train_aspect_extraction(train_dataset, val_dataset, tokenizer, num_epochs=NUM_EPOCHS, output_dir=None, resume_from_checkpoint=False, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE):
    """
    Train the Aspect Term Extraction model
    """
    if output_dir is None:
        output_dir = ASPECT_MODEL_PATH
    
    logger.info("Initializing ATE model...")
    
    # Check if we're resuming from a checkpoint
    if resume_from_checkpoint and os.path.exists(output_dir):
        logger.info(f"Loading model from {output_dir} to resume training")
        model = AutoModelForTokenClassification.from_pretrained(output_dir)
    else:
        # # Create model with higher dropout for better generalization
        # from transformers import BertConfig
        # config = BertConfig.from_pretrained(MODEL_NAME, num_labels=ASPECT_NUM_LABELS)
        
        # # Increase dropout rate for better generalization
        # config.hidden_dropout_prob = 0.3  # Default is usually 0.1
        # config.attention_probs_dropout_prob = 0.3  # Default is usually 0.1
        
        model = AutoModelForTokenClassification.from_pretrained(
            # MODEL_NAME, config=config
            MODEL_NAME, num_labels=ASPECT_NUM_LABELS
        )
    
    # Check transformers version for compatibility
    try:
        from transformers import __version__
        transformers_version = __version__
        logger.info(f"Transformers version: {transformers_version}")
    except ImportError:
        transformers_version = "unknown"
        logger.warning("Could not determine transformers version")
    
    # Configure appropriate training arguments
    # Base arguments that are common to all versions
    base_args = {
        "output_dir": f"{output_dir}/checkpoints",
        "logging_dir": f"{output_dir}/logs",
        "logging_steps": 50,
        "num_train_epochs": num_epochs,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "weight_decay": 0.01,
        "save_steps": 100,
        "eval_steps": 100,
        "save_total_limit": 3,
        "metric_for_best_model": "f1",  
        "warmup_ratio": 0.1,  # Gradual warmup for learning rate
        # # Add FP16 training for faster training if GPU supports it
        # "fp16": torch.cuda.is_available(),
        # # Add more aggressive regularization
        # "gradient_accumulation_steps": 2,  # Accumulate gradients to simulate larger batch size
    }
    
    # Add resume_from_checkpoint if needed
    if resume_from_checkpoint:
        base_args["resume_from_checkpoint"] = True
    
    # Check if eval_strategy or evaluation_strategy is the correct parameter
    train_args_signature = TrainingArguments.__init__.__code__.co_varnames
    
    if "evaluation_strategy" in train_args_signature:
        base_args["evaluation_strategy"] = "steps"
        base_args["save_strategy"] = "steps"
        base_args["load_best_model_at_end"] = True
        base_args["greater_is_better"] = True  # Higher F1 is better
        logger.info("Using 'evaluation_strategy' and 'save_strategy' parameters")
    elif "eval_strategy" in train_args_signature:
        base_args["eval_strategy"] = "steps"
        base_args["save_strategy"] = "steps"
        base_args["load_best_model_at_end"] = True
        base_args["greater_is_better"] = True  # Higher F1 is better
        logger.info("Using 'eval_strategy' and 'save_strategy' parameters")
    else:
        # Older versions might use different parameters or no specific strategy params
        logger.info("No strategy parameters found, using default configuration")
    
    # Disable wandb reporting if applicable
    if "report_to" in train_args_signature:
        base_args["report_to"] = "none"
    
    # Create training arguments with the appropriate parameters
    training_args = TrainingArguments(**base_args)
    
    # Create a data collator for token classification
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # # Create optimizer with weight decay
    # no_decay = ["bias", "LayerNorm.weight"]
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.01,
    #     },
    #     {
    #         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.0,
    #     },
    # ]
    # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    
    # Use Transformers get_scheduler for linear warmup
    from transformers import get_scheduler
    
    # Get number of training steps
    num_training_steps = num_epochs * len(train_dataset) // batch_size
    
    # First create a default scheduler with linear warmup
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),  # 10% warmup
        num_training_steps=num_training_steps
    )
    
    # Custom callback for learning rate reduction on plateau
    class F1OnPlateauCallback(TrainerCallback):
        def __init__(self, patience=12, factor=0.6, min_lr=1e-6):
            self.patience = patience
            self.factor = factor
            self.min_lr = min_lr
            self.best_f1 = -float('inf')
            self.no_improve_count = 0
        
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics is None or "eval_f1" not in metrics:
                return
            
            current_f1 = metrics["eval_f1"]
            
            if current_f1 > self.best_f1:
                self.best_f1 = current_f1
                self.no_improve_count = 0
                logger.info(f"New best F1: {current_f1:.4f}")
            else:
                self.no_improve_count += 1
                logger.info(f"F1 did not improve for {self.no_improve_count} evaluations. Current: {current_f1:.4f}, Best: {self.best_f1:.4f}")
                
                if self.no_improve_count >= self.patience:
                    # Time to reduce learning rate
                    for param_group in optimizer.param_groups:
                        old_lr = param_group['lr']
                        if old_lr > self.min_lr:
                            new_lr = max(old_lr * self.factor, self.min_lr)
                            param_group['lr'] = new_lr
                            logger.info(f"Reducing learning rate from {old_lr:.6f} to {new_lr:.6f} after {self.patience} evaluations without improvement")
                    
                    # Reset counter
                    self.no_improve_count = 0

    # Create and add our F1 plateau callback
    f1_plateau_callback = F1OnPlateauCallback(patience=15, factor=0.8, min_lr=1e-6)
    
    # Implement focal loss for better handling of class imbalance
    # class FocalLossTrainer(Trainer):
    #     def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    #         labels = inputs.get("labels")
    #         # Forward pass
    #         outputs = model(**inputs)
    #         logits = outputs.get("logits")
            
    #         # Standard loss - use model's internal loss computation
    #         if labels is None:
    #             return outputs.get("loss"), outputs
            
    #         # Custom focal loss - emphasizes hard examples
    #         gamma = 2.0  # Focus parameter - higher means more focus on hard examples
    #         alpha = 0.25  # Class balance parameter
            
    #         # Convert to one hot
    #         batch_size, seq_length, num_labels = logits.shape
    #         one_hot = torch.zeros_like(logits)
            
    #         # Only consider non-ignored positions
    #         mask = (labels >= 0).unsqueeze(-1).expand_as(one_hot)
    #         valid_labels = labels.clone()
    #         valid_labels[labels < 0] = 0  # Just for indexing, these will be masked out
            
    #         # Fill in one hot
    #         one_hot.scatter_(2, valid_labels.unsqueeze(-1), 1.0)
            
    #         # Apply focal loss calculation
    #         probs = torch.softmax(logits, dim=-1)
    #         pt = (one_hot * probs).sum(-1)  # Probability of target class
    #         pt = torch.clamp(pt, min=1e-7, max=1.0)  # Prevent NaN
            
    #         # Focal loss formula: -alpha * (1-pt)^gamma * log(pt)
    #         focal_weight = alpha * (1 - pt) ** gamma
            
    #         # Cross entropy on valid positions
    #         loss = -torch.log(pt) * focal_weight
    #         loss = loss * mask[:,:,0]  # Apply mask for ignored positions
            
    #         # Average over valid positions
    #         num_valid = mask[:,:,0].sum()
    #         if num_valid > 0:
    #             loss = loss.sum() / num_valid
    #         else:
    #             loss = loss.sum() * 0.0  # Return 0 if no valid positions
            
    #         return (loss, outputs) if return_outputs else loss

    # trainer = FocalLossTrainer(
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_ate_metrics,
        optimizers=(optimizer, lr_scheduler),  # Pass our custom optimizer and scheduler
        callbacks=[f1_plateau_callback]  # Add our custom callback
    )
    
    logger.info("Training ATE model...")
    start_time = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    end_time = time.time()
    
    training_time = end_time - start_time
    logger.info(f"ATE training completed in {training_time:.2f} seconds")
    
    # Save training metrics
    metrics = train_result.metrics
    metrics['training_time'] = training_time
    
    # Save the metrics
    with open(f"{output_dir}/training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Save the trained model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model and tokenizer saved to {output_dir}")
    
    # Evaluate the model on validation set
    logger.info("Evaluating ATE model on validation set...")
    eval_metrics = trainer.evaluate()
    
    with open(f"{output_dir}/evaluation_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=4)
    
    logger.info(f"Evaluation metrics: {eval_metrics}")
    return model, tokenizer, eval_metrics

def train_aspect_sentiment(train_dataset, val_dataset, tokenizer, num_epochs=NUM_EPOCHS, output_dir=None, resume_from_checkpoint=False, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE):
    """
    Train the Aspect Sentiment Classification model
    """
    if output_dir is None:
        output_dir = SENTIMENT_MODEL_PATH
    
    logger.info("Initializing ASC model...")
    
    # Check if we're resuming from a checkpoint
    if resume_from_checkpoint and os.path.exists(output_dir):
        logger.info(f"Loading model from {output_dir} to resume training")
        model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    else:
        # Create a model with config that includes num_labels
        from transformers import BertConfig
        config = BertConfig.from_pretrained(MODEL_NAME, num_labels=SENTIMENT_NUM_LABELS)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, config=config
        )
        
        # Manually set the class weights in the model
        model.class_weights = torch.tensor([4.0, 4.0, 1.0])
    
    # Configure appropriate training arguments
    # Base arguments that are common to all versions
    base_args = {
        "output_dir": f"{output_dir}/checkpoints",
        "logging_dir": f"{output_dir}/logs",
        "logging_steps": 50,
        "num_train_epochs": num_epochs,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "weight_decay": 0.01,
        "save_steps": 100,
        "eval_steps": 100,
        "save_total_limit": 3,
        "metric_for_best_model": "macro_f1",
        "warmup_ratio": 0.1,  # Gradual warmup for learning rate
    }
    
    # Add resume_from_checkpoint if needed
    if resume_from_checkpoint:
        base_args["resume_from_checkpoint"] = True
    
    # Check if eval_strategy or evaluation_strategy is the correct parameter
    train_args_signature = TrainingArguments.__init__.__code__.co_varnames
    
    if "evaluation_strategy" in train_args_signature:
        base_args["evaluation_strategy"] = "steps"
        base_args["save_strategy"] = "steps"
        base_args["load_best_model_at_end"] = True
        base_args["greater_is_better"] = True  # Higher F1 is better
        logger.info("Using 'evaluation_strategy' and 'save_strategy' parameters")
    elif "eval_strategy" in train_args_signature:
        base_args["eval_strategy"] = "steps"
        base_args["save_strategy"] = "steps"
        base_args["load_best_model_at_end"] = True
        base_args["greater_is_better"] = True  # Higher F1 is better
        logger.info("Using 'eval_strategy' and 'save_strategy' parameters")
    else:
        # Older versions might use different parameters or no specific strategy params
        logger.info("No strategy parameters found, using default configuration")
    
    # Disable wandb reporting if applicable
    if "report_to" in train_args_signature:
        base_args["report_to"] = "none"
    
    # Create training arguments with the appropriate parameters
    training_args = TrainingArguments(**base_args)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Create a default scheduler with linear warmup
    from transformers import get_scheduler
    
    # Get number of training steps
    num_training_steps = num_epochs * len(train_dataset) // batch_size
    
    # Create scheduler for warmup
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),  # 10% warmup
        num_training_steps=num_training_steps
    )
    
    # Custom callback for learning rate reduction on plateau
    class MacroF1OnPlateauCallback(TrainerCallback):
        def __init__(self, patience=15, factor=0.75, min_lr=1e-6):
            self.patience = patience
            self.factor = factor
            self.min_lr = min_lr
            self.best_f1 = -float('inf')
            self.no_improve_count = 0
        
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics is None or "eval_macro_f1" not in metrics:
                return
            
            current_f1 = metrics["eval_macro_f1"]
            
            if current_f1 > self.best_f1:
                self.best_f1 = current_f1
                self.no_improve_count = 0
                logger.info(f"New best Macro F1: {current_f1:.4f}")
            else:
                self.no_improve_count += 1
                logger.info(f"Macro F1 did not improve for {self.no_improve_count} evaluations. Current: {current_f1:.4f}, Best: {self.best_f1:.4f}")
                
                if self.no_improve_count >= self.patience:
                    # Time to reduce learning rate
                    for param_group in optimizer.param_groups:
                        old_lr = param_group['lr']
                        if old_lr > self.min_lr:
                            new_lr = max(old_lr * self.factor, self.min_lr)
                            param_group['lr'] = new_lr
                            logger.info(f"Reducing learning rate from {old_lr:.6f} to {new_lr:.6f} after {self.patience} evaluations without improvement")
                    
                    # Reset counter
                    self.no_improve_count = 0

    # Create and add our F1 plateau callback
    macro_f1_plateau_callback = MacroF1OnPlateauCallback(patience=15, factor=0.75, min_lr=1e-6)

    # Use standard trainer with our callbacks
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_asc_metrics,
        optimizers=(optimizer, lr_scheduler),  # Pass our custom optimizer and scheduler
        callbacks=[macro_f1_plateau_callback]  # Add our custom callback
    )
    
    logger.info("Training ASC model...")
    start_time = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    end_time = time.time()
    
    training_time = end_time - start_time
    logger.info(f"ASC training completed in {training_time:.2f} seconds")
    
    # Save training metrics
    metrics = train_result.metrics
    metrics['training_time'] = training_time
    
    # Save the metrics
    with open(f"{output_dir}/training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Save the trained model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model and tokenizer saved to {output_dir}")
    
    # Evaluate the model on validation set
    logger.info("Evaluating ASC model on validation set...")
    eval_metrics = trainer.evaluate()
    
    with open(f"{output_dir}/evaluation_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=4)
    
    logger.info(f"Evaluation metrics: {eval_metrics}")
    return model, tokenizer, eval_metrics

# ================ INFERENCE FUNCTIONS ================

class ABSAPipeline:
    """
    Pipeline for Aspect-Based Sentiment Analysis
    """
    def __init__(self, aspect_model_path=ASPECT_MODEL_PATH, sentiment_model_path=SENTIMENT_MODEL_PATH):
        # Check if the model paths exist
        if not self._check_model_exists(aspect_model_path):
            logger.error(f"Aspect model not found at {aspect_model_path}")
            raise FileNotFoundError(f"Aspect model not found at {aspect_model_path}")
            
        if not self._check_model_exists(sentiment_model_path):
            logger.error(f"Sentiment model not found at {sentiment_model_path}")
            raise FileNotFoundError(f"Sentiment model not found at {sentiment_model_path}")
            
        logger.info(f"Loading models from {aspect_model_path} and {sentiment_model_path}")
        
        try:
            self.aspect_tokenizer = AutoTokenizer.from_pretrained(aspect_model_path)
            self.aspect_model = AutoModelForTokenClassification.from_pretrained(aspect_model_path)
            
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path)
            
            # Create NER pipeline for aspect extraction
            self.aspect_pipeline = pipeline(
                "ner", 
                model=self.aspect_model, 
                tokenizer=self.aspect_tokenizer,
                aggregation_strategy="simple"
            )
            
            # Create classification pipeline for sentiment analysis
            self.sentiment_pipeline = pipeline(
                "text-classification", 
                model=self.sentiment_model, 
                tokenizer=self.sentiment_tokenizer
            )
            
            logger.info("ABSA Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise
    
    def _check_model_exists(self, path):
        """Check if model files exist at the specified path"""
        if os.path.exists(path):
            config_file = os.path.join(path, "config.json")
            model_file = False
            
            # Check for either pytorch_model.bin or model files in the form of model.safetensors
            if os.path.exists(os.path.join(path, "pytorch_model.bin")):
                model_file = True
            else:
                for file in os.listdir(path):
                    if file.startswith("model") and (file.endswith(".bin") or file.endswith(".safetensors")):
                        model_file = True
                        break
            
            return os.path.exists(config_file) and model_file
        return False
    
    def extract_aspects(self, text, confidence_threshold=0.05):
        """Extract aspect terms from text using a more direct approach with a lower confidence threshold"""
        if not text or not text.strip():
            logger.warning("Empty text provided for aspect extraction")
            return []
        
        try:
            # Tokenize the input
            inputs = self.aspect_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            
            # Move to GPU if available
            device = next(self.aspect_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get the predictions
            with torch.no_grad():
                outputs = self.aspect_model(**inputs)
                predictions = outputs.logits
            
            # Convert to probabilities
            probs = torch.nn.functional.softmax(predictions, dim=2)
            # Get the predicted labels
            pred_labels = torch.argmax(probs, dim=2)
            
            # Convert to numpy for easier handling
            probs_np = probs.detach().cpu().numpy()[0]
            pred_labels_np = pred_labels.detach().cpu().numpy()[0]
            
            # Get tokens
            tokens = self.aspect_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].cpu())
            
            # Extract aspects with confidence above threshold
            aspects = []
            i = 0
            while i < len(tokens):
                if tokens[i] in ['[CLS]', '[SEP]', '[PAD]']:
                    i += 1
                    continue
                    
                if pred_labels_np[i] == ASPECT_LABEL_MAP['B-ASP'] and probs_np[i][pred_labels_np[i]] >= confidence_threshold:
                    # Found the beginning of an aspect
                    aspect_start = i
                    aspect_end = i
                    aspect_score = float(probs_np[i][pred_labels_np[i]])
                    
                    # Look for continuation (I-ASP)
                    j = i + 1
                    while j < len(tokens) and j < len(pred_labels_np):
                        if tokens[j] in ['[CLS]', '[SEP]', '[PAD]']:
                            j += 1
                            continue
                            
                        # Consider subword tokens as part of the aspect
                        if (pred_labels_np[j] == ASPECT_LABEL_MAP['I-ASP'] and probs_np[j][pred_labels_np[j]] >= confidence_threshold) or tokens[j].startswith('##'):
                            aspect_end = j
                            j += 1
                        else:
                            break
                    
                    # Extract the aspect text
                    aspect_tokens = tokens[aspect_start:aspect_end+1]
                    # Remove ## from subword tokens and join
                    aspect_text = ''.join([t.replace('##', '') for t in aspect_tokens])
                    
                    # Filter out very short aspects (1-2 characters)
                    if len(aspect_text) <= 2:
                        i = aspect_end + 1
                        continue
                    
                    # Calculate character offsets in the original text
                    # Note: This is an approximation as tokenization can be complex
                    original_tokens = text.split()
                    start_char = -1
                    end_char = -1
                    
                    # Simple search for the aspect in original text
                    aspect_text_lower = aspect_text.lower()
                    text_lower = text.lower()
                    start_char = text_lower.find(aspect_text_lower)
                    if start_char >= 0:
                        end_char = start_char + len(aspect_text)
                    
                    aspects.append({
                        'word': aspect_text,
                        'score': aspect_score,
                        'start': start_char,
                        'end': end_char
                    })
                    
                    i = aspect_end + 1
                else:
                    i += 1
            
            return aspects
        except Exception as e:
            logger.error(f"Error in aspect extraction: {str(e)}")
            return []
    
    def classify_sentiment(self, text, aspect):
        """Classify sentiment for a given text-aspect pair"""
        if not text or not aspect:
            logger.warning("Empty text or aspect provided for sentiment classification")
            return {'label': 1, 'score': 0.0}  # Default to neutral
        
        try:
            # Tokenize the input
            inputs = self.sentiment_tokenizer(text, aspect, return_tensors="pt", truncation=True, padding=True)
            
            # Move to GPU if available
            device = next(self.sentiment_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get the predictions
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                predictions = outputs.logits
            
            # Convert to probabilities
            probs = torch.nn.functional.softmax(predictions, dim=1)
            # Get the predicted label
            pred_label = torch.argmax(probs, dim=1).item()
            # Get the confidence score
            confidence = float(probs[0][pred_label])
            
            return {
                'label': pred_label,
                'score': confidence
            }
        except Exception as e:
            logger.error(f"Error in sentiment classification: {str(e)}")
            return {'label': 1, 'score': 0.0}  # Default to neutral
    
    def analyze(self, text):
        """Full ABSA pipeline: extract aspects and determine their sentiment"""
        logger.info(f"Analyzing text: {text}")
        
        if not text or not text.strip():
            logger.warning("Empty text provided for analysis")
            return []
        
        # Extract aspects
        aspects = self.extract_aspects(text)
        
        if not aspects:
            logger.info("No aspects found in the text")
            return []
            
        logger.info(f"Found {len(aspects)} aspects")
        
        # Analyze sentiment for each aspect
        results = []
        for aspect in aspects:
            aspect_term = aspect['word']
            sentiment = self.classify_sentiment(text, aspect_term)
            
            results.append({
                'aspect': aspect_term,
                'sentiment': SENTIMENT_LABELS[sentiment['label']],
                'sentiment_score': sentiment['score'],
                'aspect_score': aspect['score'],
                'start': aspect['start'],
                'end': aspect['end']
            })
        
        return results

# ================ EVALUATION FUNCTIONS ================

def load_and_process_example(file_path, pipeline, num_examples=5):
    """Process examples from a test file using the pipeline"""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []
        
    try:
        # Load the JSONL file (one JSON object per line)
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    data.append(json.loads(line))
        
        if not data:
            logger.warning(f"No data found in {file_path}")
            return []
            
        # Take just a few examples for demonstration
        if len(data) > num_examples:
            data = data[:num_examples]
        
        results = []
        for item in data:
            text = item.get('text', '')
            if not text:
                logger.warning("Skipping item without text field")
                continue
                
            gold_aspects = [asp_info.get('aspect', '') for asp_info in item.get('aspects', [])]
            gold_aspects = [aspect for aspect in gold_aspects if aspect]  # Filter out empty aspects
            
            # Run the pipeline
            predictions = pipeline.analyze(text)
            
            # Compare with gold standard aspects
            result = {
                'text': text,
                'gold_aspects': gold_aspects,
                'predicted_aspects': predictions
            }
            results.append(result)
        
        return results
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return []

def evaluate_predictions(results):
    """Simple evaluation of predictions vs. gold standard"""
    if not results:
        return {'precision': 0, 'recall': 0, 'f1': 0, 'num_examples': 0}
        
    total_gold = 0
    total_pred = 0
    total_correct = 0
    
    for result in results:
        gold_aspects = set(result['gold_aspects'])
        pred_aspects = set([p['aspect'] for p in result['predicted_aspects']])
        
        # Count exact matches (could be improved with fuzzy matching)
        correct = len(gold_aspects.intersection(pred_aspects))
        
        total_gold += len(gold_aspects)
        total_pred += len(pred_aspects)
        total_correct += correct
    
    # Calculate metrics
    precision = total_correct / total_pred if total_pred > 0 else 0
    recall = total_correct / total_gold if total_gold > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'num_examples': len(results)
    }

# ================ TEST TRAINING FUNCTIONS ================

def test_train_aspect_extraction(train_dataset, val_dataset, tokenizer, num_epochs=1, output_dir=None):
    """
    Train the Aspect Term Extraction model for just one epoch (testing purposes)
    """
    if output_dir is None:
        output_dir = f"{SAVED_MODELS_DIR}/aspect_extractor_model_test"
    
    logger.info("Initializing ATE model for test training...")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=ASPECT_NUM_LABELS
    )
    
    # Configure appropriate training arguments
    # Base arguments that are common to all versions
    base_args = {
        "output_dir": f"{output_dir}/checkpoints",
        "logging_dir": f"{output_dir}/logs",
        "logging_steps": 5,  # Log more frequently for test
        "num_train_epochs": num_epochs,
        "learning_rate": LEARNING_RATE,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "weight_decay": 0.01,
        "save_steps": 10,
        "eval_steps": 10,
        "save_total_limit": 2,
    }
    
    # Check if eval_strategy or evaluation_strategy is the correct parameter
    train_args_signature = TrainingArguments.__init__.__code__.co_varnames
    
    if "evaluation_strategy" in train_args_signature:
        base_args["evaluation_strategy"] = "steps"
        base_args["save_strategy"] = "steps"
        logger.info("Using 'evaluation_strategy' and 'save_strategy' parameters")
    elif "eval_strategy" in train_args_signature:
        base_args["eval_strategy"] = "steps"
        base_args["save_strategy"] = "steps"
        logger.info("Using 'eval_strategy' and 'save_strategy' parameters")
    else:
        # Older versions might use different parameters or do_eval flag
        base_args["do_eval"] = True
        logger.info("Using 'do_eval' parameter")
    
    # Disable wandb reporting if applicable
    if "report_to" in train_args_signature:
        base_args["report_to"] = "none"
    
    # Create training arguments with the appropriate parameters
    training_args = TrainingArguments(**base_args)
    
    # Create a data collator for token classification
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_ate_metrics
    )
    
    logger.info("Test training ATE model for one epoch...")
    start_time = time.time()
    train_result = trainer.train()
    end_time = time.time()
    
    training_time = end_time - start_time
    logger.info(f"ATE test training completed in {training_time:.2f} seconds")
    
    # Evaluate the model
    logger.info("Evaluating test ATE model...")
    eval_metrics = trainer.evaluate()
    
    logger.info(f"Test evaluation metrics: {eval_metrics}")
    
    # Save the model and tokenizer to the specified directory
    logger.info(f"Saving test model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save evaluation metrics
    with open(f"{output_dir}/evaluation_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=4)
    
    return model, tokenizer, eval_metrics

def test_train_aspect_sentiment(train_dataset, val_dataset, tokenizer, num_epochs=1, output_dir=None):
    """
    Train the Aspect Sentiment Classification model for just one epoch (testing purposes)
    """
    if output_dir is None:
        output_dir = f"{SAVED_MODELS_DIR}/aspect_sentiment_model_test"
    
    logger.info("Initializing ASC model for test training...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=SENTIMENT_NUM_LABELS
    )
    
    # Configure appropriate training arguments
    # Base arguments that are common to all versions
    base_args = {
        "output_dir": f"{output_dir}/checkpoints",
        "logging_dir": f"{output_dir}/logs",
        "logging_steps": 5,  # Log more frequently for test
        "num_train_epochs": num_epochs,
        "learning_rate": LEARNING_RATE,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "weight_decay": 0.01,
        "save_steps": 10,
        "eval_steps": 10,
        "save_total_limit": 2,
    }
    
    # Check if eval_strategy or evaluation_strategy is the correct parameter
    train_args_signature = TrainingArguments.__init__.__code__.co_varnames
    
    if "evaluation_strategy" in train_args_signature:
        base_args["evaluation_strategy"] = "steps"
        base_args["save_strategy"] = "steps"
        logger.info("Using 'evaluation_strategy' and 'save_strategy' parameters")
    elif "eval_strategy" in train_args_signature:
        base_args["eval_strategy"] = "steps"
        base_args["save_strategy"] = "steps"
        logger.info("Using 'eval_strategy' and 'save_strategy' parameters")
    else:
        # Older versions might use different parameters or do_eval flag
        base_args["do_eval"] = True
        logger.info("Using 'do_eval' parameter")
    
    # Disable wandb reporting if applicable
    if "report_to" in train_args_signature:
        base_args["report_to"] = "none"
    
    # Create training arguments with the appropriate parameters
    training_args = TrainingArguments(**base_args)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_asc_metrics
    )
    
    logger.info("Test training ASC model for one epoch...")
    start_time = time.time()
    train_result = trainer.train()
    end_time = time.time()
    
    training_time = end_time - start_time
    logger.info(f"ASC test training completed in {training_time:.2f} seconds")
    
    # Evaluate the model
    logger.info("Evaluating test ASC model...")
    eval_metrics = trainer.evaluate()
    
    logger.info(f"Test evaluation metrics: {eval_metrics}")
    
    # Save the model and tokenizer to the specified directory
    logger.info(f"Saving test model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save evaluation metrics
    with open(f"{output_dir}/evaluation_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=4)
    
    return model, tokenizer, eval_metrics

# ================ INITIALIZATION FUNCTION ================

def initialize_tokenizer():
    """Initialize tokenizer from the pretrained model"""
    logger.info(f"Initializing tokenizer from {MODEL_NAME}")
    return AutoTokenizer.from_pretrained(MODEL_NAME)
