import json
import time
import os
import torch
import numpy as np
import logging
import re
import unicodedata
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForTokenClassification
from transformers import pipeline
from transformers import TrainerCallback, EarlyStoppingCallback
from datasets import Dataset
from seqeval.metrics import classification_report
from sklearn.metrics import classification_report as cls_report
from torch.nn import functional as F
import torch.nn as nn

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) 

# Constants
MODEL_NAME = "xlm-roberta-base"
SAVED_MODELS_DIR = "models/saved_models_xlm"
NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
MAX_LENGTH = 128  # Maximum sequence length
PATIENCE = 30     # Patience for early stopping if plateau reached at min LR

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

# Preprocessing function for XLM-RoBERTa (minimal preprocessing since the model handles various languages)
def preprocess_text(text):
    """
    Minimal preprocessing for XLM-RoBERTa:
    - convert to lowercase
    - basic cleaning
    """
    if not text:
        return ""
    text = str(text).lower().strip()
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    return text

def enhanced_align_tokens_and_labels(original_tokens, original_labels, tokenizer):
    """
    Improved alignment of original tokens and their BIO labels with XLM-RoBERTa tokenizer output.
    This handles the subword tokenization with better error checking and robustness.
    
    Args:
        original_tokens: List of original tokens from the dataset
        original_labels: List of BIO labels corresponding to original tokens
        tokenizer: The tokenizer
        
    Returns:
        List of aligned labels for tokenized input (including special tokens)
    """
    # Start with special tokens
    aligned_tokens = []
    aligned_labels = []
    
    # Add start token (XLM-RoBERTa uses <s>)
    aligned_tokens.append("<s>")
    aligned_labels.append("O")  # Start token is always outside
    
    # Process each original token and align with tokenized subwords
    for orig_token, orig_label in zip(original_tokens, original_labels):
        # Skip empty tokens
        if not orig_token:
            continue
            
        # Minimal preprocessing for XLM-RoBERTa
        cleaned_token = preprocess_text(orig_token)
        
        # Tokenize the original token to get subwords
        subwords = tokenizer.tokenize(cleaned_token)
        
        # If no subwords were produced (rare, but could happen with some tokens)
        if not subwords:
            # Try with the original token as fallback
            subwords = tokenizer.tokenize(orig_token)
            if not subwords:
                # If still no subwords, skip this token
                continue
        
        # Add the subwords and their labels
        for i, subword in enumerate(subwords):
            aligned_tokens.append(subword)
            
            # First subword gets the original label
            if i == 0:
                aligned_labels.append(orig_label)
            else:
                # For subsequent subwords:
                # - If original was B-ASP, subsequent is I-ASP
                # - If original was I-ASP, subsequent remains I-ASP
                # - If original was O, subsequent remains O
                if orig_label == "B-ASP":
                    aligned_labels.append("I-ASP")
                else:
                    aligned_labels.append(orig_label)
    
    # Add end token (XLM-RoBERTa uses </s>)
    aligned_tokens.append("</s>")
    aligned_labels.append("O")  # End token is always outside
    
    # Verify the alignment
    if len(aligned_tokens) != len(aligned_labels):
        logger.warning(f"Mismatch in aligned tokens ({len(aligned_tokens)}) and labels ({len(aligned_labels)})")
        # Attempt to fix by truncating to the shorter length
        min_len = min(len(aligned_tokens), len(aligned_labels))
        aligned_tokens = aligned_tokens[:min_len]
        aligned_labels = aligned_labels[:min_len]
    
    return aligned_tokens, aligned_labels

def convert_aligned_labels_to_ids(aligned_labels, label_map):
    """Convert string labels to IDs using the label map"""
    return [label_map.get(label, 0) for label in aligned_labels]

# ================ CRF LAYER IMPLEMENTATION ================
# Has extra computational cost, but improves performance 
# A CRF layer models the dependencies between labels in the sequence — it can learn, for example, 
# that I-Aspect can only follow B-Aspect or another I-Aspect, never an O.

class CRF(torch.nn.Module):
    """
    Conditional Random Field implementation for sequence labeling tasks
    """
    def __init__(self, num_tags, batch_first=True):
        super(CRF, self).__init__()
        
        self.num_tags = num_tags
        self.batch_first = batch_first
        
        # Transition matrix: [num_tags, num_tags]
        # transitions[i, j] represents the score of transitioning from tag i to tag j
        self.transitions = torch.nn.Parameter(torch.randn(num_tags, num_tags))
        
        # Initialize special start and end transitions
        # These constraints are based on BIO tagging scheme:
        # - Cannot transition from OUTSIDE to INSIDE (O -> I-ASP) - make this very unlikely
        # - Can transition from anything to OUTSIDE (? -> O) and from OUTSIDE to BEGIN (O -> B-ASP)
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        
        # Make transition from O (0) to I-ASP (2) very unlikely
        self.transitions.data[0, 2] = -10.0
        
    def forward(self, emissions, tags, mask=None, reduction='mean'):
        """
        Calculate negative log likelihood (NLL) loss for the CRF
        
        Args:
            emissions: [batch_size, seq_len, num_tags] emission score tensor
            tags: [batch_size, seq_len] ground truth tags
            mask: [batch_size, seq_len] mask tensor, 1 for valid positions, 0 for padding
            
        Returns:
            loss: scalar, negative log likelihood loss
            sequence of predicted tags
        """
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8, device=tags.device)
            
        if self.batch_first:
            # Convert to [seq_len, batch_size, num_tags]
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)
            
        batch_size = emissions.size(1)
        seq_len = emissions.size(0)
        
        # Calculate log-likelihood score for the provided tags
        score = self._score_sentence(emissions, tags, mask)
        
        # Calculate the partition function (normalization term)
        forward_score = self._forward_algorithm(emissions, mask)
        
        # Loss = normalization term - tag sequence score
        loss = forward_score - score
        
        # Return mean loss across batch
        if reduction == 'none':
            return loss
        elif reduction == 'sum':
            return loss.sum()
        else:  # Default is 'mean'
            return loss.mean()
    
    def decode(self, emissions, mask=None):
        """
        Find the most likely tag sequence using Viterbi algorithm
        
        Args:
            emissions: [batch_size, seq_len, num_tags] emission score tensor
            mask: [batch_size, seq_len] mask tensor, 1 for valid positions, 0 for padding
            
        Returns:
            tags: [batch_size, seq_len] best tag sequence
        """
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.uint8, device=emissions.device)
            
        if self.batch_first:
            # Convert to [seq_len, batch_size, num_tags]
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)
            
        return self._viterbi_decode(emissions, mask)
    
    def _score_sentence(self, emissions, tags, mask):
        """
        Calculate the score for a given tag sequence
        
        Args:
            emissions: [seq_len, batch_size, num_tags]
            tags: [seq_len, batch_size]
            mask: [seq_len, batch_size]
            
        Returns:
            score: [batch_size], sequence score for each batch
        """
        seq_len, batch_size = tags.shape
        
        # Initialize score
        score = torch.zeros(batch_size, device=emissions.device)
        
        # Add first emission score for the first tag
        first_tag = tags[0]
        score += emissions[0, torch.arange(batch_size), first_tag]
        
        # Add remaining emission scores and transition scores
        for i in range(1, seq_len):
            # Previous and current tags
            prev_tag = tags[i-1]
            curr_tag = tags[i]
            
            # Add emission score for current tag
            score += emissions[i, torch.arange(batch_size), curr_tag] * mask[i]
            
            # Add transition score from previous to current tag
            transition_score = self.transitions[prev_tag, curr_tag]
            score += transition_score * mask[i]
            
        return score
    
    def _forward_algorithm(self, emissions, mask):
        """
        Calculate the partition function using forward algorithm (dynamic programming)
        
        Args:
            emissions: [seq_len, batch_size, num_tags]
            mask: [seq_len, batch_size]
            
        Returns:
            log_partition: [batch_size], log partition for each batch
        """
        seq_len, batch_size, num_tags = emissions.shape
        
        # Initialize forward variables
        alphas = torch.full((batch_size, num_tags), -10000.0, device=emissions.device)
        
        # Start with everything is possible
        alphas[:, 0] = 0  # Start with O tag (index 0)
        
        for i in range(seq_len):
            # [batch_size, 1, num_tags] -> [batch_size, num_tags, 1]
            emit_scores = emissions[i].unsqueeze(-1)
            
            # [batch_size, num_tags, 1] + [num_tags, num_tags] -> [batch_size, num_tags, num_tags]
            next_tag_scores = alphas.unsqueeze(-1) + self.transitions
            
            # [batch_size, num_tags]
            next_tag_scores = torch.logsumexp(next_tag_scores, dim=1)
            
            # Update alpha values with emission scores
            mask_i = mask[i].unsqueeze(-1)
            alphas = mask_i * (next_tag_scores + emit_scores.squeeze(-1)) + (1 - mask_i) * alphas
            
        # Return log partition function
        log_partition = torch.logsumexp(alphas, dim=-1)
        return log_partition
        
    def _viterbi_decode(self, emissions, mask):
        """
        Decode the best tag sequence using Viterbi algorithm
        
        Args:
            emissions: [seq_len, batch_size, num_tags]
            mask: [seq_len, batch_size]
            
        Returns:
            best_tags: [batch_size, seq_len], best tag sequence for each batch
        """
        seq_len, batch_size, num_tags = emissions.shape
        
        # Initialize variables
        # score at step i (for each tag)
        scores = torch.zeros(batch_size, num_tags, device=emissions.device)
        
        # backpointer to previous best tag
        history = torch.zeros((seq_len, batch_size, num_tags), dtype=torch.long, device=emissions.device)
        
        # At step 0, the previous tag is just the tag itself
        # scores = scores + emissions[0]
        
        # For each step
        for i in range(seq_len):
            # If it's the first step
            if i == 0:
                scores = scores + emissions[0]
                continue
                
            # broadcast previous scores
            broadcast_scores = scores.unsqueeze(-1)  # [batch_size, num_tags, 1]
            
            # calculate transition scores for all possible next tags
            transition_scores = broadcast_scores + self.transitions  # [batch_size, num_tags, num_tags]
            
            # find the best previous tag for each current tag
            max_scores, history[i] = torch.max(transition_scores, dim=1)  # [batch_size, num_tags]
            
            # add emission scores for current tags
            scores = max_scores + emissions[i] * mask[i].unsqueeze(-1)
            
        # Find the best final score and corresponding tag
        max_final_scores, max_final_tags = torch.max(scores, dim=1)  # [batch_size]
        
        # Backtrack to find the best path
        best_tags = torch.zeros((batch_size, seq_len), dtype=torch.long, device=emissions.device)
        best_tags[:, -1] = max_final_tags
        
        # Backtrack through the history to find the best tag sequence
        for i in range(seq_len-2, -1, -1):
            best_tags[:, i] = torch.gather(history[i+1], 1, best_tags[:, i+1].unsqueeze(1)).squeeze(1)
            
        # Convert to batch-first format if needed
        if self.batch_first:
            best_tags = best_tags.transpose(0, 1)
            
        return best_tags

# Custom model with CRF layer
class XLMRobertaForTokenClassificationCRF(torch.nn.Module):
    """
    XLM-RoBERTa model with token classification head and CRF layer
    """
    def __init__(self, model_name, num_labels, crf_dropout=0.1):
        super(XLMRobertaForTokenClassificationCRF, self).__init__()
        
        # Load base model
        self.roberta = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
        
        # Add CRF layer
        self.crf = CRF(num_labels, batch_first=True)
        
        # Add dropout before CRF
        self.dropout = torch.nn.Dropout(crf_dropout)
        
        # Set config attributes
        self.config = self.roberta.config
        self.num_labels = num_labels
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        """
        Forward pass with CRF on top
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (not used for XLM-RoBERTa)
            labels: Labels for computing loss
            
        Returns:
            loss: CRF loss if labels are provided
            logits: Raw emission scores
            predictions: CRF decoded tag sequence if not training
        """
        # Get XLM-RoBERTa output
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )
        
        # Get logits from the token classification head
        logits = outputs.logits
        
        # Apply dropout
        logits = self.dropout(logits)
        
        # Initialize outputs dictionary
        output = {
            "logits": logits,
        }
        
        # If labels are provided, compute CRF loss
        if labels is not None:
            # Compute CRF loss
            loss = self.crf(logits, labels, mask=attention_mask)
            output["loss"] = loss
        
        # If not training, decode the best tag sequence
        if not self.training:
            predictions = self.crf.decode(logits, mask=attention_mask)
            output["predictions"] = predictions
        
        return output

# ================ DATASET LOADING FUNCTIONS ================

def load_aspect_keywords(file_path="data/aspect_keywords_map.json"):
    """Load aspect keywords map to help the model focus on relevant terms"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            aspect_keywords_map = json.load(f)
        
        # Preprocess all keywords
        for aspect, keywords in aspect_keywords_map.items():
            aspect_keywords_map[aspect] = [preprocess_text(k) for k in keywords if preprocess_text(k)]
            
        # Create a flattened list of all keywords
        all_keywords = []
        for keywords in aspect_keywords_map.values():
            all_keywords.extend(keywords)
            
        # Remove duplicates
        all_keywords = list(set(all_keywords))
        
        logger.info(f"Loaded {len(all_keywords)} unique aspect keywords")
        return aspect_keywords_map, all_keywords
    except Exception as e:
        logger.error(f"Error loading aspect keywords: {e}")
        return {}, []

# Load keywords once at module level
ASPECT_KEYWORDS_MAP, ALL_ASPECT_KEYWORDS = load_aspect_keywords()

def load_aspect_dataset(file_path, tokenizer, label_map=ASPECT_LABEL_MAP):
    """
    Load and preprocess the ATE (Aspect Term Extraction) dataset for XLM-RoBERTa
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
        if 'text' not in item or not item['text']:
            logger.warning(f"Skipping item without text: {item}")
            continue
            
        text = item['text']
        
        # Check if we have pre-generated tokens and BIO labels
        if 'tokens' in item and 'bio_labels' in item:
            orig_tokens = item['tokens']
            orig_bio_labels = item['bio_labels']
            
            # Align the original tokens and labels with XLM-RoBERTa tokenization
            aligned_tokens, aligned_labels = enhanced_align_tokens_and_labels(orig_tokens, orig_bio_labels, tokenizer)
            
            # Convert aligned labels to IDs
            label_ids = convert_aligned_labels_to_ids(aligned_labels, label_map)
            
            # Tokenize the text directly for XLM-RoBERTa
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
            
            # Verify all values are integers
            if not all(isinstance(x, int) for x in encoding['input_ids']):
                logger.warning(f"Input IDs contain non-integer values: {encoding['input_ids'][:10]}...")
                continue
                
            if not all(isinstance(x, int) for x in encoding['attention_mask']):
                logger.warning(f"Attention mask contains non-integer values: {encoding['attention_mask'][:10]}...")
                continue
                
            if not all(isinstance(x, int) for x in label_ids):
                logger.warning(f"Labels contain non-integer values: {label_ids[:10]}...")
                continue
                
            # Create entry with consistent keys - excluding offset_mapping
            entry = {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'labels': label_ids
            }
            formatted_data.append(entry)
        else:
            # For data without pre-tokenized text and labels, just use text
            encoding = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors=None  # Return Python lists
            )
            
            # Initialize all tokens with 'O' label (0)
            labels = [0] * len(encoding['input_ids'])
            
            # Verify all values are integers
            if not all(isinstance(x, int) for x in encoding['input_ids']):
                logger.warning(f"Input IDs contain non-integer values: {encoding['input_ids'][:10]}...")
                continue
                
            if not all(isinstance(x, int) for x in encoding['attention_mask']):
                logger.warning(f"Attention mask contains non-integer values: {encoding['attention_mask'][:10]}...")
                continue
            
            entry = {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'labels': labels
            }
            formatted_data.append(entry)

    logger.info(f"Loaded {len(formatted_data)} examples from {file_path}")
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

# ================ CUSTOM TRAINER WITH CRF SUPPORT ================

class CRFTrainer(Trainer):
    """
    Custom Trainer class to handle models with CRF layer
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom loss computation for CRF model
        
        Args:
            model: The model to train
            inputs: The inputs to the model
            return_outputs: Whether to return the outputs along with the loss
            **kwargs: Additional keyword arguments (ignored)
        
        Returns:
            loss: The loss value
            outputs: The model outputs (if return_outputs=True)
        """
        labels = inputs.pop("labels")
        outputs = model(
            input_ids=inputs["input_ids"], 
            attention_mask=inputs["attention_mask"], 
            labels=labels
        )
        
        loss = outputs["loss"] if "loss" in outputs else None
        
        if return_outputs:
            return loss, outputs
        return loss
    
    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        """
        Custom prediction step for CRF model
        """
        inputs = self._prepare_inputs(inputs)
        
        # Make a copy of inputs to avoid modifying the original
        inputs_copy = {k: v for k, v in inputs.items()}
        
        # Keep only the required keys for the model
        required_keys = ["input_ids", "attention_mask"]
        for key in list(inputs_copy.keys()):
            if key not in required_keys and key != "labels":
                inputs_copy.pop(key, None)
        
        with torch.no_grad():
            # Forward pass
            labels = inputs_copy.pop("labels", None)
            outputs = model(
                input_ids=inputs_copy["input_ids"], 
                attention_mask=inputs_copy["attention_mask"]
            )
            
            # Get predictions from CRF
            if "predictions" in outputs:
                preds = outputs["predictions"]
            else:
                # Fall back to taking argmax over logits
                preds = torch.argmax(outputs["logits"], dim=-1)
            
            # Get loss
            loss = None
            if not prediction_loss_only and labels is not None:
                # Compute loss with labels
                loss_outputs = model(
                    input_ids=inputs_copy["input_ids"], 
                    attention_mask=inputs_copy["attention_mask"], 
                    labels=labels
                )
                loss = loss_outputs["loss"] if "loss" in loss_outputs else None
            
            # Prepare logits for metrics computation
            logits = outputs["logits"]
        
        if prediction_loss_only:
            return (loss, None, None)
        
        return (loss, logits, labels)

# ================ CUSTOM LEARNING RATE SCHEDULER CALLBACK ================

class CustomLearningRateSchedulerCallback(TrainerCallback):
    """
    Custom callback to adjust learning rate based on a specified metric plateau
    and handle early stopping once min_lr is reached and no improvement is observed.
    """
    def __init__(self, optimizer, scheduler, metric_name, patience=10, factor=0.5, min_lr=1e-9, stopping_patience=30):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metric_name = metric_name  # e.g., "eval_f1" or "eval_macro_f1"
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.stopping_patience = stopping_patience
        
        self.best_metric_value = -float('inf')
        self.no_improve_count = 0
        self.min_lr_reached = False
        self.no_improve_since_min_lr = 0 # Counter for evaluations since min_lr was reached
        
        self.scheduler_manually_controlled = False # Flag to indicate if we've taken over LR scheduling
        self.original_scheduler_step = None

    def _noop_scheduler_step(self, *args, **kwargs): # Accept any args trainer might pass
        """A no-operation step function to replace the original scheduler's step."""
        pass

    def _take_manual_control_of_scheduler(self):
        """Disables the trainer's default scheduler behavior by replacing its step method."""
        if not self.scheduler_manually_controlled and self.scheduler is not None:
            logger.info(f"Callback taking manual control of LR from scheduler {type(self.scheduler).__name__}.")
            self.original_scheduler_step = self.scheduler.step
            self.scheduler.step = self._noop_scheduler_step
            self.scheduler_manually_controlled = True
            
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or self.metric_name not in metrics:
            logger.warning(f"Metric '{self.metric_name}' not found in evaluation metrics: {metrics}. Skipping LR adjustment.")
            return

        current_metric_value = metrics[self.metric_name]
        current_epoch = state.epoch
        # Ensure optimizer.param_groups is not empty and has 'lr'
        if not self.optimizer.param_groups or 'lr' not in self.optimizer.param_groups[0]:
            logger.warning("Optimizer param_groups empty or 'lr' not found. Skipping LR adjustment.")
            return
        current_lr = self.optimizer.param_groups[0]['lr']

        if current_metric_value > self.best_metric_value:
            self.best_metric_value = current_metric_value
            self.no_improve_count = 0
            if self.min_lr_reached: # Reset this counter if improvement happens even after min_lr
                self.no_improve_since_min_lr = 0
            logger.info(f"New best {self.metric_name}: {current_metric_value:.4f} at epoch {current_epoch:.2f}. LR: {current_lr:.2e}")
        else:
            self.no_improve_count += 1
            if self.min_lr_reached:
                self.no_improve_since_min_lr += 1
            
            logger.info(
                f"{self.metric_name}: {current_metric_value:.4f} (LR: {current_lr:.2e}). "
                f"No improvement for {self.no_improve_count} evals. Best: {self.best_metric_value:.4f}."
            )

            if self.min_lr_reached and self.no_improve_since_min_lr >= self.stopping_patience:
                logger.info(
                    f"Early stopping: Min LR ({self.min_lr:.2e}) reached and no improvement on {self.metric_name} "
                    f"for {self.no_improve_since_min_lr} (>= {self.stopping_patience}) evaluations."
                )
                control.should_training_stop = True
                return

            if self.no_improve_count >= self.patience:
                if current_lr <= self.min_lr: # Already at or below min_lr
                    if not self.min_lr_reached:
                        logger.info(f"Reached minimum learning rate of {self.min_lr:.2e}. Monitoring for stopping_patience.")
                        self.min_lr_reached = True
                    # Do not reduce further, just wait for stopping_patience evaluations
                else: # Reduce LR
                    self._take_manual_control_of_scheduler() 
                    
                    new_lr = max(current_lr * self.factor, self.min_lr)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    
                    logger.info(f"Reducing LR: {current_lr:.2e} -> {new_lr:.2e} (factor={self.factor}) due to {self.metric_name} plateau.")
                    self.no_improve_count = 0 # Reset patience counter after reduction
                    
                    if new_lr <= self.min_lr:
                        if not self.min_lr_reached:
                             logger.info(f"Reached minimum learning rate of {self.min_lr:.2e} after reduction. Monitoring for stopping_patience.")
                        self.min_lr_reached = True
                        self.no_improve_since_min_lr = 0 # Reset counter for stopping_patience

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Event called after logging the last logs."""
        if self.scheduler_manually_controlled and logs is not None and 'learning_rate' in logs:
            # Remove the potentially stale 'learning_rate' from trainer's default log
            # as our callback is managing it and logging the accurate one separately.
            del logs['learning_rate']

# ================ DATA AUGMENTATION WITH BACK-TRANSLATION ================

def back_translate(text, source_lang="el", target_lang="en", max_length=128):
    """
    Augment data using back-translation
    
    This would typically use a translation API or model like MarianMT.
    For now, we'll just return the original text as a placeholder.
    
    Args:
        text: Original text to be back-translated
        source_lang: Source language code
        target_lang: Target language code
        max_length: Maximum sequence length
        
    Returns:
        Augmented text after back-translation (currently just returns original text)
    """
    # For testing purposes, just return the original text
    logger.info("Back-translation is disabled - returning original text")
    return text

# ================ MODEL TRAINING FUNCTIONS ================

def find_latest_complete_checkpoint(checkpoints_dir):
    """
    Find the latest complete checkpoint directory that contains all necessary files.
    
    Args:
        checkpoints_dir: Path to the checkpoints directory
        
    Returns:
        Path to the latest complete checkpoint directory, or None if none found
    """
    if not os.path.exists(checkpoints_dir):
        return None
    
    # Get all checkpoint directories
    checkpoint_dirs = []
    for item in os.listdir(checkpoints_dir):
        if item.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoints_dir, item)):
            try:
                # Extract checkpoint number for sorting
                checkpoint_num = int(item.split("-")[1])
                checkpoint_dirs.append((checkpoint_num, os.path.join(checkpoints_dir, item)))
            except (IndexError, ValueError):
                logger.warning(f"Could not parse checkpoint number from directory: {item}")
                continue
    
    # Sort by checkpoint number (descending - latest first)
    checkpoint_dirs.sort(key=lambda x: x[0], reverse=True)
    
    # Check each checkpoint directory for completeness
    # For Hugging Face Trainer checkpoints, "trainer_state.json" and a model file are key.
    # Model file can be "pytorch_model.bin" or "model.safetensors".
    required_files_options = [
        ["trainer_state.json", "pytorch_model.bin"],
        ["trainer_state.json", "model.safetensors"]
    ]
    
    for checkpoint_num, checkpoint_path in checkpoint_dirs:
        is_complete = False
        for required_set in required_files_options:
            files_present = all(os.path.exists(os.path.join(checkpoint_path, f)) for f in required_set)
            if files_present:
                is_complete = True
                break
        
        if is_complete:
            logger.info(f"Found complete checkpoint: {checkpoint_path}")
            return checkpoint_path
        else:
            logger.warning(f"Checkpoint {checkpoint_path} is incomplete - missing required files (e.g., trainer_state.json and model file).")
            
    logger.warning(f"No complete checkpoints found in {checkpoints_dir}")
    return None

def train_aspect_extraction(train_dataset, val_dataset, tokenizer, num_epochs=NUM_EPOCHS, output_dir=None, 
                          resume_from_checkpoint_cli_flag=False, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, 
                          use_crf=True, patience=PATIENCE, gradient_clipping=1.0):
    """
    Train the Aspect Term Extraction model using XLM-RoBERTa with CRF layer
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        tokenizer: Tokenizer to use
        num_epochs: Number of training epochs
        output_dir: Directory to save the model
        resume_from_checkpoint_cli_flag: Whether to resume from an existing checkpoint (CLI flag)
        learning_rate: Learning rate for training
        batch_size: Batch size for training
        use_crf: Whether to use CRF layer
        patience: Patience for early stopping after reaching min learning rate (in evaluations)
        gradient_clipping: Gradient clipping parameter
    """
    if output_dir is None:
        output_dir = ASPECT_MODEL_PATH
    
    logger.info("Initializing ATE model training...")

    actual_checkpoint_to_resume = None
    checkpoints_subdir_path = os.path.join(output_dir, "checkpoints")
    logs_subdir_path = os.path.join(output_dir, "logs")

    if resume_from_checkpoint_cli_flag:
        if os.path.exists(checkpoints_subdir_path):
            actual_checkpoint_to_resume = find_latest_complete_checkpoint(checkpoints_subdir_path)
        if actual_checkpoint_to_resume:
            logger.info(f"Will resume training from checkpoint: {actual_checkpoint_to_resume}")
        else:
            logger.warning(
                f"Resume from checkpoint was requested, but no complete checkpoint was found in {checkpoints_subdir_path}. "
                "Training will start from scratch. Output directory will not be deleted."
            )
    else: # resume_from_checkpoint_cli_flag is False
        logger.info(f"Starting a new training run (resume_from_checkpoint_cli_flag=False).")
        if os.path.exists(output_dir):
            logger.info(f"Deleting existing model directory: {output_dir}")
            import shutil
            try:
                shutil.rmtree(output_dir)
                logger.info(f"Successfully deleted {output_dir}.")
            except OSError as e:
                logger.error(f"Error deleting directory {output_dir}: {e}. Training will proceed, but old files might interfere.")
    
    # Ensure directories exist for training
    os.makedirs(checkpoints_subdir_path, exist_ok=True)
    os.makedirs(logs_subdir_path, exist_ok=True)
    
    # Initialize model structure. Trainer will load from actual_checkpoint_to_resume if set.
    if use_crf:
        logger.info(f"Initializing XLM-RoBERTa with CRF layer (model structure).")
        model = XLMRobertaForTokenClassificationCRF(MODEL_NAME, ASPECT_NUM_LABELS)
    else:
        logger.info(f"Initializing standard XLM-RoBERTa for token classification (model structure).")
        from transformers import XLMRobertaConfig
        config = XLMRobertaConfig.from_pretrained(MODEL_NAME, num_labels=ASPECT_NUM_LABELS)
        config.hidden_dropout_prob = 0.3
        config.attention_probs_dropout_prob = 0.3
        model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, config=config)
    
    # Configure appropriate training arguments
    train_args_dict = {
        "output_dir": checkpoints_subdir_path, # Checkpoints saved here
        "logging_dir": logs_subdir_path,       # Logs saved here
        "logging_steps": 60,
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "weight_decay": 0.01,
        "save_steps": 120, # Consider making this configurable
        "eval_steps": 120, # Consider making this configurable
        "save_total_limit": 3,
        "metric_for_best_model": "f1",
        "eval_strategy": "steps",
        "load_best_model_at_end": True,
        "greater_is_better": True,
        "report_to": "none",
        "remove_unused_columns": False,
        "max_grad_norm": gradient_clipping
    }
    
    training_args = TrainingArguments(**train_args_dict)
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    logger.info(f"Using AdamW optimizer with learning rate: {learning_rate}")
    
    from transformers import get_scheduler
    try:
        num_update_steps_per_epoch = len(train_dataset) // batch_size
        if num_update_steps_per_epoch == 0 : # Handle small datasets
             num_update_steps_per_epoch = 1
        num_training_steps = num_epochs * num_update_steps_per_epoch
        if num_training_steps == 0:
            logger.warning("Number of training steps is zero. Scheduler might not work as expected.")
            num_training_steps = 1 # Avoid division by zero for warmup
    except TypeError:
        logger.warning("Could not determine train_dataset length for scheduler steps, using a large default.")
        num_training_steps = 10000 # Fallback, consider if this is appropriate

    warmup_steps = int(0.1 * num_training_steps)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    logger.info(f"Using linear warmup scheduler with {warmup_steps} warmup steps over {num_training_steps} total steps.")
    
    optimizer_ref = optimizer
    scheduler_ref = lr_scheduler

    lr_reduction_patience_ate = max(2, patience // 2)
    early_stop_patience_ate = 4 * patience

    custom_lr_scheduler_callback = CustomLearningRateSchedulerCallback(
        optimizer=optimizer_ref,
        scheduler=scheduler_ref,
        metric_name="eval_f1",
        patience=lr_reduction_patience_ate,
        factor=0.5,
        min_lr=1e-6,
        stopping_patience=early_stop_patience_ate
    )
    
    logger.info(f"ATE LR reduction patience: {lr_reduction_patience_ate}, Factor: 0.5, Early stopping patience: {early_stop_patience_ate}")
    
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=early_stop_patience_ate,
        early_stopping_threshold=0.001
    )
    
    trainer_class = CRFTrainer if use_crf else Trainer
    
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_ate_metrics,
        optimizers=(optimizer, lr_scheduler),
        callbacks=[custom_lr_scheduler_callback, early_stopping_callback]
    )
    
    logger.info("Starting ATE model training run...")
    start_time = time.time()
    # Pass the specific checkpoint path (or None) to trainer.train()
    train_result = trainer.train(resume_from_checkpoint=actual_checkpoint_to_resume)
    end_time = time.time()
    
    training_time = end_time - start_time
    logger.info(f"ATE training completed in {training_time:.2f} seconds")
    
    metrics = train_result.metrics
    metrics['training_time'] = training_time
    
    # Save metrics to the main output_dir (not checkpoints_subdir_path)
    with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    # HuggingFace Trainer with load_best_model_at_end=True will save the best model
    # to training_args.output_dir (checkpoints_subdir_path/best_model or similar)
    # and then copy it to the top-level output_dir if not using custom save.
    # The .save_pretrained() call below ensures the final model (which is the best due to load_best_model_at_end)
    # is saved to the main output_dir.
    
    logger.info(f"Saving final model (best model from training) to {output_dir}")
    if use_crf:
        # Save the CRF model's state_dict and config to the main output_dir
        # The trainer already saved the best checkpoint. We want the final state in output_dir.
        # If model is already the best model (due to load_best_model_at_end), we can save it directly.
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        model.config.save_pretrained(output_dir)
    else:
        model.save_pretrained(output_dir)
        
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model and tokenizer saved to {output_dir}")
    
    logger.info("Evaluating ATE model on validation set (using the best model loaded at end of training)...")
    eval_metrics = trainer.evaluate()
    
    with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(eval_metrics, f, indent=4)
    
    logger.info(f"Evaluation metrics: {eval_metrics}")
    return model, tokenizer, eval_metrics

# ================ CHARACTER OFFSET RECOVERY FUNCTIONS ================

def recover_char_offsets(tokens, predictions, offset_mapping=None, text=None):
    """
    Recover character-level offsets for predicted aspect terms.
    
    Args:
        tokens: List of tokens
        predictions: List of predicted label IDs
        offset_mapping: List of (start, end) tuples for each token (optional)
        text: Original text (optional)
        
    Returns:
        List of dictionaries with aspect term, start and end character positions
    """
    aspects = []
    
    # If we don't have offset_mapping, we can only return token-level predictions
    if offset_mapping is None:
        # Track whether we're inside an aspect
        in_aspect = False
        current_aspect = []
        
        for token_idx, pred in enumerate(predictions):
            # Handle B-ASP tag - start of a new aspect
            if pred == ASPECT_LABEL_MAP["B-ASP"]:
                # If we were already in an aspect, add it
                if in_aspect and current_aspect:
                    aspects.append({
                        "text": " ".join(current_aspect),
                        "start": -1,  # No character offsets available
                        "end": -1
                    })
                    
                # Start new aspect
                in_aspect = True
                current_aspect = [tokens[token_idx] if token_idx < len(tokens) else "[UNK]"]
            
            # Handle I-ASP tag - continue current aspect
            elif pred == ASPECT_LABEL_MAP["I-ASP"] and in_aspect:
                if token_idx < len(tokens):
                    current_aspect.append(tokens[token_idx])
            
            # Handle O tag - end current aspect if we were in one
            elif in_aspect:
                aspects.append({
                    "text": " ".join(current_aspect),
                    "start": -1,  # No character offsets available
                    "end": -1
                })
                in_aspect = False
                current_aspect = []
        
        # Don't forget the last aspect if there is one
        if in_aspect and current_aspect:
            aspects.append({
                "text": " ".join(current_aspect),
                "start": -1,  # No character offsets available
                "end": -1
            })
            
        return aspects
    
    # If we have offset_mapping, we can get character-level offsets
    # Track whether we're inside an aspect
    in_aspect = False
    aspect_start = -1
    aspect_end = -1
    
    for token_idx, (pred, (char_start, char_end)) in enumerate(zip(predictions, offset_mapping)):
        # Skip special tokens (they have (0,0) offset)
        if char_start == 0 and char_end == 0:
            continue
            
        # Handle B-ASP tag - start of a new aspect
        if pred == ASPECT_LABEL_MAP["B-ASP"]:
            # If we were already in an aspect, add it
            if in_aspect:
                aspect_text = text[aspect_start:aspect_end] if text else "[UNKNOWN]"
                aspects.append({
                    "text": aspect_text,
                    "start": aspect_start,
                    "end": aspect_end
                })
                
            # Start new aspect
            in_aspect = True
            aspect_start = char_start
            aspect_end = char_end
        
        # Handle I-ASP tag - continue current aspect
        elif pred == ASPECT_LABEL_MAP["I-ASP"] and in_aspect:
            aspect_end = char_end
        
        # Handle O tag - end current aspect if we were in one
        elif in_aspect:
            aspect_text = text[aspect_start:aspect_end] if text else "[UNKNOWN]"
            aspects.append({
                "text": aspect_text,
                "start": aspect_start,
                "end": aspect_end
            })
            in_aspect = False
    
    # Don't forget the last aspect if there is one
    if in_aspect:
        aspect_text = text[aspect_start:aspect_end] if text else "[UNKNOWN]"
        aspects.append({
            "text": aspect_text,
            "start": aspect_start,
            "end": aspect_end
        })
    
    return aspects

# ================ INFERENCE FUNCTIONS ================

class ABSAPipeline:
    """
    Pipeline for Aspect-Based Sentiment Analysis using XLM-RoBERTa model with CRF layer
    """
    def __init__(self, aspect_model_path=ASPECT_MODEL_PATH, sentiment_model_path=SENTIMENT_MODEL_PATH, use_crf=True):
        # Check if the model paths exist
        if not self._check_model_exists(aspect_model_path):
            logger.error(f"Aspect model not found at {aspect_model_path}")
            raise FileNotFoundError(f"Aspect model not found at {aspect_model_path}")
            
        logger.info(f"Loading aspect extraction model from {aspect_model_path}")
        
        try:
            self.aspect_tokenizer = AutoTokenizer.from_pretrained(aspect_model_path)
            
            if use_crf:
                # Load CRF model
                self.aspect_model = XLMRobertaForTokenClassificationCRF(MODEL_NAME, ASPECT_NUM_LABELS)
                # Load state dict
                self.aspect_model.load_state_dict(torch.load(os.path.join(aspect_model_path, "pytorch_model.bin")))
            else:
                # Use standard HuggingFace model
                self.aspect_model = AutoModelForTokenClassification.from_pretrained(aspect_model_path)
            
            # Load sentiment model if path exists
            if self._check_model_exists(sentiment_model_path):
                logger.info(f"Loading sentiment classification model from {sentiment_model_path}")
                self.sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)
                self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path)
                self.has_sentiment_model = True
            else:
                logger.warning(f"Sentiment model not found at {sentiment_model_path}. Only aspect extraction will be available.")
                self.has_sentiment_model = False
            
            # Move to GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.aspect_model.to(self.device)
            self.aspect_model.eval()
            
            if self.has_sentiment_model:
                self.sentiment_model.to(self.device)
                self.sentiment_model.eval()
            
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

    def extract_aspects(self, text, use_crf=True):
        """Extract aspect terms from text with character-level offsets"""
        if not text or not text.strip():
            logger.warning("Empty text provided for aspect extraction")
            return []
        
        try:
            # Preprocess text minimally
            text_proc = preprocess_text(text)
            
            # Tokenize the input
            encoding = self.aspect_tokenizer(
                text_proc, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=MAX_LENGTH,
                return_offsets_mapping=True
            )
            
            # Get offset mapping before moving tensors to device
            offset_mapping = encoding.pop('offset_mapping').squeeze().tolist()
            
            # Move to device
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            
            # Get the predictions
            with torch.no_grad():
                if use_crf:
                    # Use CRF model for prediction
                    outputs = self.aspect_model(**encoding)
                    if "predictions" in outputs:
                        predictions = outputs["predictions"].squeeze().cpu().numpy()
                    else:
                        # Fall back to logits
                        logits = outputs["logits"]
                        predictions = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
                else:
                    # Use standard model
                    outputs = self.aspect_model(**encoding)
                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
            
            # Recover character-level offsets for the predicted aspects
            aspects = recover_char_offsets(
                self.aspect_tokenizer.convert_ids_to_tokens(encoding['input_ids'].squeeze().tolist()),
                predictions,
                offset_mapping,
                text
            )
            
            # Add confidence score (placeholder for now)
            for aspect in aspects:
                aspect["score"] = 1.0
            
            return aspects
        except Exception as e:
            logger.error(f"Error in aspect extraction: {str(e)}")
            return []

    def classify_sentiment(self, text, aspect):
        """Classify sentiment for a given text-aspect pair"""
        if not self.has_sentiment_model:
            logger.warning("Sentiment model not loaded. Cannot classify sentiment.")
            return {'label': 1, 'score': 0.0}  # Default to neutral
            
        if not text or not aspect:
            logger.warning("Empty text or aspect provided for sentiment classification")
            return {'label': 1, 'score': 0.0}  # Default to neutral
        
        try:
            # Preprocess text and aspect
            text_proc = preprocess_text(text)
            aspect_proc = preprocess_text(aspect)
            
            # Tokenize the input as a text-aspect pair
            encoding = self.sentiment_tokenizer(
                text_proc, 
                aspect_proc, 
                return_tensors="pt", 
                truncation=True, 
                padding=True,
                max_length=MAX_LENGTH
            )
            
            # Move to device
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            
            # Get the predictions
            with torch.no_grad():
                outputs = self.sentiment_model(**encoding)
                logits = outputs.logits
                
                # Convert to probabilities
                probs = torch.nn.functional.softmax(logits, dim=1)
                
                # Get the predicted label and confidence
                pred_label = torch.argmax(probs, dim=1).item()
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
        
        # Analyze sentiment for each aspect if sentiment model is available
        results = []
        for aspect in aspects:
            aspect_term = aspect['text']
            
            # Classify sentiment if model is available
            if self.has_sentiment_model:
                sentiment = self.classify_sentiment(text, aspect_term)
                sentiment_label = SENTIMENT_LABELS[sentiment['label']]
                sentiment_score = sentiment['score']
            else:
                # Default to neutral if no sentiment model
                sentiment_label = "neutral"
                sentiment_score = 0.0
            
            results.append({
                'aspect': aspect_term,
                'sentiment': sentiment_label,
                'sentiment_score': sentiment_score,
                'aspect_score': aspect['score'],
                'start': aspect['start'],
                'end': aspect['end']
            })
        
        return results

# ================ INITIALIZATION FUNCTION ================

def initialize_tokenizer():
    """Initialize tokenizer from the pretrained model"""
    logger.info(f"Initializing tokenizer from {MODEL_NAME}")
    return AutoTokenizer.from_pretrained(MODEL_NAME)

# Add load_sentiment_dataset function
def load_sentiment_dataset(file_path, tokenizer):
    """
    Load and preprocess the ASC (Aspect Sentiment Classification) dataset for XLM-RoBERTa
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
        # Check if this is direct aspect-sentiment data (text, aspect, sentiment_id format)
        if 'text' in item and 'aspect' in item and 'sentiment_id' in item:
            text = item['text']
            aspect = item['aspect']
            sentiment_id = item['sentiment_id']
            
            # Preprocess text and aspect
            text_proc = preprocess_text(text)
            aspect_proc = preprocess_text(aspect)
            
            # Encode the text and aspect as a pair
            encoding = tokenizer(
                text_proc, 
                aspect_proc, 
                padding="max_length", 
                truncation=True, 
                max_length=MAX_LENGTH,
                return_tensors=None  # Return Python lists
            )
            
            # Create entry
            entry = {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'labels': sentiment_id
            }
            formatted_data.append(entry)
        
        # Check for traditional format with nested aspects
        elif 'text' in item and 'aspects' in item:
            text = item['text']
            # Create entries for each aspect and its sentiment
            for aspect_info in item['aspects']:
                aspect = aspect_info.get('aspect', '')
                if not aspect:
                    continue
                
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
                
                # Preprocess text and aspect
                text_proc = preprocess_text(text)
                aspect_proc = preprocess_text(aspect)
                
                # Encode the text and aspect as a pair
                encoding = tokenizer(
                    text_proc, 
                    aspect_proc, 
                    padding="max_length", 
                    truncation=True, 
                    max_length=MAX_LENGTH,
                    return_tensors=None  # Return Python lists
                )
                
                # Create entry
                entry = {
                    'input_ids': encoding['input_ids'],
                    'attention_mask': encoding['attention_mask'],
                    'labels': sentiment_id
                }
                formatted_data.append(entry)
    
    return Dataset.from_list(formatted_data)

# Add function to compute ASC metrics
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

# Add train_aspect_sentiment function
def train_aspect_sentiment(
    train_dataset, 
    val_dataset, 
    tokenizer, 
    num_epochs=NUM_EPOCHS, 
    output_dir=None, 
    resume_from_checkpoint_cli_flag=False, 
    learning_rate=LEARNING_RATE, 
    batch_size=BATCH_SIZE, 
    patience=PATIENCE, 
    gradient_clipping=1.0, 
    class_weights=None,
    lr_reduction_patience_asc=None
):
    """
    Train the Aspect Sentiment Classification model using XLM-RoBERTa
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        tokenizer: Tokenizer to use
        num_epochs: Number of training epochs
        output_dir: Directory to save the model
        resume_from_checkpoint_cli_flag: Whether to resume from an existing checkpoint (CLI flag)
        learning_rate: Learning rate for training
        batch_size: Batch size for training
        patience: Patience for early stopping after reaching min learning rate (in evaluations)
        gradient_clipping: Gradient clipping value
        class_weights: Optional class weights tensor for handling class imbalance
        lr_reduction_patience_asc: Optional learning rate reduction patience for ASC training
    """
    if output_dir is None:
        output_dir = SENTIMENT_MODEL_PATH
    
    logger.info("Initializing ASC model training...")

    actual_checkpoint_to_resume = None
    checkpoints_subdir_path = os.path.join(output_dir, "checkpoints")
    logs_subdir_path = os.path.join(output_dir, "logs")

    if resume_from_checkpoint_cli_flag:
        if os.path.exists(checkpoints_subdir_path):
            actual_checkpoint_to_resume = find_latest_complete_checkpoint(checkpoints_subdir_path)
        if actual_checkpoint_to_resume:
            logger.info(f"Will resume training from checkpoint: {actual_checkpoint_to_resume}")
        else:
            logger.warning(
                f"Resume from checkpoint was requested, but no complete checkpoint was found in {checkpoints_subdir_path}. "
                "Training will start from scratch. Output directory will not be deleted."
            )
    else: # resume_from_checkpoint_cli_flag is False
        logger.info(f"Starting a new training run (resume_from_checkpoint_cli_flag=False).")
        if os.path.exists(output_dir):
            logger.info(f"Deleting existing model directory: {output_dir}")
            import shutil
            try:
                shutil.rmtree(output_dir)
                logger.info(f"Successfully deleted {output_dir}.")
            except OSError as e:
                logger.error(f"Error deleting directory {output_dir}: {e}. Training will proceed, but old files might interfere.")

    # Ensure directories exist for training
    os.makedirs(checkpoints_subdir_path, exist_ok=True)
    os.makedirs(logs_subdir_path, exist_ok=True)

    # Initialize model structure. Trainer will load from actual_checkpoint_to_resume if set.
    logger.info(f"Initializing standard XLM-RoBERTa for sequence classification (model structure).")
    from transformers import XLMRobertaConfig
    config = XLMRobertaConfig.from_pretrained(MODEL_NAME, num_labels=SENTIMENT_NUM_LABELS)
    config.hidden_dropout_prob = 0.15 # ASC specific dropout - Aligned to ATE's non-CRF
    config.attention_probs_dropout_prob = 0.15 # ASC specific dropout - Aligned to ATE's non-CRF
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
        
    # Configure appropriate training arguments
    train_args_dict = {
        "output_dir": checkpoints_subdir_path, # Checkpoints saved here
        "logging_dir": logs_subdir_path,       # Logs saved here
        "logging_steps": 60, # Aligned with ATE
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "weight_decay": 0.01,
        "save_steps": 120, # Aligned with ATE
        "eval_steps": 120, # Aligned with ATE
        "save_total_limit": 3,
        "metric_for_best_model": "macro_f1",
        "eval_strategy": "steps",
        "load_best_model_at_end": True,
        "greater_is_better": True,
        "report_to": "none",
        "remove_unused_columns": False,
        "max_grad_norm": gradient_clipping
    }
    
    # Note: The original code had "resume_from_checkpoint": True in train_args if CLI flag was set.
    # This is handled by passing actual_checkpoint_to_resume to trainer.train() instead.
    
    training_args = TrainingArguments(**train_args_dict)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # Aligned with ATE's optimizer
    logger.info(f"Using AdamW optimizer with learning rate: {learning_rate}")
    
    from transformers import get_scheduler
    try:
        num_update_steps_per_epoch = len(train_dataset) // batch_size
        if num_update_steps_per_epoch == 0 : # Handle small datasets
             num_update_steps_per_epoch = 1
        num_training_steps = num_epochs * num_update_steps_per_epoch
        if num_training_steps == 0:
            logger.warning("Number of training steps is zero. Scheduler might not work as expected.")
            num_training_steps = 1 
    except TypeError:
        logger.warning("Could not determine train_dataset length for scheduler steps, using a large default.")
        num_training_steps = 10000

    warmup_steps = int(0.1 * num_training_steps)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    logger.info(f"Using linear warmup scheduler with {warmup_steps} warmup steps over {num_training_steps} total steps.")
    
    optimizer_ref = optimizer
    scheduler_ref = lr_scheduler

    if lr_reduction_patience_asc is None:
        lr_reduction_patience_asc = max(2, patience // 2)
    early_stop_patience_asc = 3 * patience # Aligned with ATE logic

    custom_lr_scheduler_callback = CustomLearningRateSchedulerCallback(
        optimizer=optimizer_ref,
        scheduler=scheduler_ref,
        metric_name="eval_macro_f1",
        patience=lr_reduction_patience_asc,
        factor=0.5,
        min_lr=1e-9,
        stopping_patience=early_stop_patience_asc 
    )
    
    logger.info(f"ASC LR reduction patience: {lr_reduction_patience_asc}, Factor: 0.5, Early stopping patience: {early_stop_patience_asc}")
    
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=early_stop_patience_asc,
        early_stopping_threshold=0.001
    )

    # Custom loss for handling class imbalance with support for class weights
    class WeightedFocalLossTrainer(Trainer):
        def __init__(self, class_weights=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights
            if class_weights is not None:
                logger.info(f"Using class weights in loss computation: {class_weights}")
            else:
                logger.info("Using standard focal loss without class weights")
        
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            
            if self.class_weights is not None:
                # Use weighted cross entropy with focal loss
                # Move class weights to the same device as logits
                weights = self.class_weights.to(logits.device)
                
                # Apply focal loss with per-class alpha (aligned with class weights)
                gamma = 3.0  # Focus parameter
                alpha = weights  # Use class weights as per-class alpha
                
                # Compute cross entropy with class weights
                ce_loss = F.cross_entropy(logits, labels, weight=weights, reduction='none')
                
                # Compute focal weights
                probs = torch.softmax(logits, dim=1)
                pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
                focal_weights = alpha[labels] * (1 - pt) ** gamma  # Index alpha by labels
                
                # Apply focal weights
                loss = focal_weights * ce_loss
                loss = loss.mean()
            else:
                # Standard focal loss without class weights
                gamma = 3.0
                num_labels = len(SENTIMENT_LABELS)
                one_hot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1.0)
                probs = torch.nn.functional.softmax(logits, dim=1)
                pt = (one_hot * probs).sum(dim=1)
                focal_weights = (1 - pt) ** gamma
                per_sample_losses = -torch.log(pt + 1e-10) * focal_weights
                loss = per_sample_losses.mean()
            
            return (loss, outputs) if return_outputs else loss
    
    # Create trainer with weighted focal loss for better handling of class imbalance
    trainer = WeightedFocalLossTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_asc_metrics,
        optimizers=(optimizer, lr_scheduler),
        callbacks=[custom_lr_scheduler_callback, early_stopping_callback]
    )
    
    logger.info("Starting ASC model training run...")
    start_time = time.time()
    # Pass the specific checkpoint path (or None) to trainer.train()
    train_result = trainer.train(resume_from_checkpoint=actual_checkpoint_to_resume)
    end_time = time.time()
    
    training_time = end_time - start_time
    logger.info(f"ASC training completed in {training_time:.2f} seconds")
    
    metrics = train_result.metrics
    metrics['training_time'] = training_time
    
    with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Saving final model (best model from training) to {output_dir}")
    model.save_pretrained(output_dir) # Standard save for sequence classification model
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model and tokenizer saved to {output_dir}")
    
    logger.info("Evaluating ASC model on validation set (using the best model loaded at end of training)...")
    eval_metrics = trainer.evaluate()
    
    with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(eval_metrics, f, indent=4)
    
    logger.info(f"Evaluation metrics: {eval_metrics}")
    return model, tokenizer, eval_metrics
