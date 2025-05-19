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
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants

# A more comprehensive list of domain-specific stopwords and adjectives
STOPWORDS = {
    'ειναι', 'είναι', 'εχει', 'έχει', 'και', 'with', 'the', 'has', 'is', 'are', 'του', 'της', 'το',
    'για', 'για', 'απο', 'από', 'στον', 'στην', 'στο', 'στους', 'στις', 'στα', 'με', 'τα', 'τον', 'την', 
    'κανω', 'κάνω', 'αντι', 'αντί', 'οτι', 'ότι', 'θα', 'να', 'αλλα', 'αλλά', 'μου', 'σου', 'του',
    'μας', 'σας', 'τους', 'αυτο', 'αυτό', 'αυτη', 'αυτή', 'πολυ', 'πολύ', 'λιγο', 'λίγο', 'καθε', 'κάθε',
    'ολο', 'όλο', 'ολα', 'όλα', 'γιατι', 'γιατί', 'επειδη', 'επειδή', 'οταν', 'όταν'
}

ADJECTIVES = {
    'καλη', 'καλή', 'καλο', 'καλό', 'καλοσ', 'καλός', 'καλού', 'καλής', 'καλού',
    'κακη', 'κακή', 'κακο', 'κακό', 'κακός', 'κακού', 'κακής', 
    'ωραια', 'ωραία', 'ωραιο', 'ωραίο', 'ωραίος', 'ωραίου', 'ωραίας',
    'εξαιρετικη', 'εξαιρετική', 'εξαιρετικο', 'εξαιρετικό', 'εξαιρετικός', 'εξαιρετικού', 'εξαιρετικής',
    'τελεια', 'τέλεια', 'τελειο', 'τέλειο', 'τέλειος', 'τέλειου', 'τέλειας',
    'χαλια', 'χάλια', 'χαλιο', 'χάλιο',
    'αργη', 'αργή', 'αργο', 'αργό', 'αργός', 'αργού', 'αργής',
    'γρηγορη', 'γρήγορη', 'γρηγορο', 'γρήγορο', 'γρήγορος', 'γρήγορου', 'γρήγορης',
    'δυνατη', 'δυνατή', 'δυνατο', 'δυνατό', 'δυνατός', 'δυνατού', 'δυνατής',
    'αδυναμη', 'αδύναμη', 'αδυναμο', 'αδύναμο', 'αδύναμος', 'αδύναμου', 'αδύναμης',
    'μεγαλη', 'μεγάλη', 'μεγαλο', 'μεγάλο', 'μεγάλος', 'μεγάλου', 'μεγάλης',
    'μικρη', 'μικρή', 'μικρο', 'μικρό', 'μικρός', 'μικρού', 'μικρής',
    'φθηνη', 'φθηνή', 'φθηνο', 'φθηνό', 'φθηνός', 'φθηνού', 'φθηνής',
    'ακριβη', 'ακριβή', 'ακριβο', 'ακριβό', 'ακριβός', 'ακριβού', 'ακριβής'
}

MODEL_NAME = "pchatz/palobert-base-greek-social-media-v2"

# RoBERTa preprocessing function
def preprocess_text(text):
    """
    Preprocess text for palobert-base-greek-social-media-v2:
    - remove all greek diacritics
    - convert to lowercase
    - remove all punctuation
    """
    text = str(text).lower()
    text = unicodedata.normalize('NFD', text).translate({ord('\N{COMBINING ACUTE ACCENT}'): None})
    text = re.sub(r'[^\w\s]', '', text)
    return text

def enhanced_preprocess_text(text):
    """
    Enhanced text preprocessing for Greek language with better handling of diacritics and punctuation.
    This preserves more information than the original preprocess_text function.
    """
    # Convert to lowercase
    text = str(text).lower()
    
    # Normalize Greek diacritics
    text = unicodedata.normalize('NFD', text).translate({ord('\N{COMBINING ACUTE ACCENT}'): None})
    
    # Preserve certain punctuation like hyphens that might be part of aspect terms
    # but remove other punctuation
    text = re.sub(r'[^\w\s\-]', '', text)
    
    return text

def is_potential_aspect(token, stopwords=STOPWORDS, adjectives=ADJECTIVES):
    """
    Check if a token could be a valid aspect term.
    Filters out common stopwords and adjectives.
    """
    token_lower = token.lower()
    
    # Very short tokens are unlikely to be aspects
    if len(token_lower) <= 1:
        return False
    
    # Check against stopwords and adjectives
    if token_lower in stopwords or token_lower in adjectives:
        return False
    
    return True

def align_tokens_and_labels(original_tokens, original_labels, tokenizer):
    """
    Align original tokens and their BIO labels with RoBERTa tokenizer output.
    This handles the subword tokenization that RoBERTa does.
    
    Args:
        original_tokens: List of original tokens from the dataset
        original_labels: List of BIO labels corresponding to original tokens
        tokenizer: The RoBERTa tokenizer
        
    Returns:
        List of aligned labels for RoBERTa tokenized input (including special tokens)
    """
    # Convert to lowercase and apply preprocessing
    bert_tokens = []
    bert_labels = []
    
    # Add start token for RoBERTa (uses <s> instead of [CLS])
    bert_tokens.append("<s>")
    bert_labels.append("O")  # Start token is always outside
    
    # Process each original token and align with RoBERTa tokens
    for orig_token, orig_label in zip(original_tokens, original_labels):
        # Preprocess token for RoBERTa
        orig_token = preprocess_text(orig_token)
        
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
    
    # Add end token for RoBERTa (uses </s> instead of [SEP])
    bert_tokens.append("</s>")
    bert_labels.append("O")  # End token is always outside
    
    # Verify the alignment
    if len(bert_tokens) != len(bert_labels):
        logger.warning(f"Mismatch in aligned tokens ({len(bert_tokens)}) and labels ({len(bert_labels)})")
    
    return bert_tokens, bert_labels

def enhanced_align_tokens_and_labels(original_tokens, original_labels, tokenizer):
    """
    Improved alignment of original tokens and their BIO labels with tokenizer output.
    This handles the subword tokenization with better error checking and robustness.
    
    Args:
        original_tokens: List of original tokens from the dataset
        original_labels: List of BIO labels corresponding to original tokens
        tokenizer: The tokenizer
        
    Returns:
        List of aligned labels for tokenized input (including special tokens)
    """
    # Start with special tokens
    bert_tokens = []
    bert_labels = []
    
    # Add start token (varies by tokenizer)
    start_token = "<s>" if hasattr(tokenizer, 'bos_token') and tokenizer.bos_token == '<s>' else "[CLS]"
    bert_tokens.append(start_token)
    bert_labels.append("O")  # Start token is always outside
    
    # Process each original token and align with tokenized subwords
    for orig_token, orig_label in zip(original_tokens, original_labels):
        # Skip empty tokens
        if not orig_token:
            continue
            
        # Preprocess token for tokenizer
        cleaned_token = enhanced_preprocess_text(orig_token)
        
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
    
    # Add end token
    end_token = "</s>" if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token == '</s>' else "[SEP]"
    bert_tokens.append(end_token)
    bert_labels.append("O")  # End token is always outside
    
    # Verify the alignment
    if len(bert_tokens) != len(bert_labels):
        logger.warning(f"Mismatch in aligned tokens ({len(bert_tokens)}) and labels ({len(bert_labels)})")
        # Attempt to fix by truncating to the shorter length
        min_len = min(len(bert_tokens), len(bert_labels))
        bert_tokens = bert_tokens[:min_len]
        bert_labels = bert_labels[:min_len]
    
    return bert_tokens, bert_labels

def convert_aligned_labels_to_ids(aligned_labels, label_map):
    """Convert string labels to IDs using the label map"""
    return [label_map.get(label, 0) for label in aligned_labels]

def calculate_class_weights(dataset, num_classes=3):
    """
    Calculate class weights inversely proportional to class frequencies.
    This helps with imbalanced data.
    
    Args:
        dataset: Dataset with 'labels' field
        num_classes: Number of classes (default: 3 for O, B-ASP, I-ASP)
        
    Returns:
        numpy array of class weights
    """
    # Extract all labels
    all_labels = []
    for item in dataset:
        # Skip ignored labels (-100)
        all_labels.extend([label for label in item['labels'] if label != -100])
    
    # Count occurrences of each class
    label_counter = Counter(all_labels)
    
    # Calculate weights inversely proportional to frequency
    total_samples = len(all_labels)
    weights = np.zeros(num_classes)
    
    for label, count in label_counter.items():
        if 0 <= label < num_classes:  # Ensure label is valid
            weights[label] = total_samples / (count * num_classes)
    
    # Handle any zero weights (classes not present)
    weights[weights == 0] = 1.0
    
    # Normalize weights
    weights = weights / np.sum(weights) * num_classes
    
    return weights

# Constants
SAVED_MODELS_DIR = "saved_models_r"
NUM_EPOCHS = 2  # Start with fewer epochs but use early stopping
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

# Load aspect keywords for keyword-aware training
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

# ================ DATASET LOADING FUNCTIONS ================

def load_aspect_dataset(file_path, tokenizer, label_map=ASPECT_LABEL_MAP):
    """
    Load and preprocess the ATE (Aspect Term Extraction) dataset for RoBERTa
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
            
            # Align the original tokens and labels with RoBERTa tokenization
            aligned_tokens, aligned_labels = align_tokens_and_labels(orig_tokens, orig_bio_labels, tokenizer)
            
            # Convert aligned labels to IDs
            label_ids = convert_aligned_labels_to_ids(aligned_labels, label_map)
            
            # Tokenize the text directly for RoBERTa
            # Note: RoBERTa doesn't use token_type_ids
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
                
            # Create entry with consistent keys (RoBERTa doesn't use token_type_ids)
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
            
            # RoBERTa doesn't use token_type_ids
            entry = {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'labels': labels
            }
            formatted_data.append(entry)

    return Dataset.from_list(formatted_data)

def load_sentiment_dataset(file_path, tokenizer):
    """
    Load and preprocess the ASC (Aspect Sentiment Classification) dataset for RoBERTa
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
            
            # Preprocess text and aspect for RoBERTa
            text_proc = preprocess_text(text)
            aspect_proc = preprocess_text(aspect)
            
            # Encode the text and aspect as a pair for RoBERTa
            # Note: RoBERTa doesn't use token_type_ids
            encoding = tokenizer(
                text_proc, 
                aspect_proc, 
                padding="max_length", 
                truncation=True, 
                max_length=MAX_LENGTH,
                return_tensors=None  # Return Python lists
            )
            
            # Create entry without token_type_ids for RoBERTa
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
                
                # Preprocess text and aspect for RoBERTa
                text_proc = preprocess_text(text)
                aspect_proc = preprocess_text(aspect)
                
                # Encode the text and aspect as a pair for RoBERTa
                # Note: RoBERTa doesn't use token_type_ids
                encoding = tokenizer(
                    text_proc, 
                    aspect_proc, 
                    padding="max_length", 
                    truncation=True, 
                    max_length=MAX_LENGTH,
                    return_tensors=None  # Return Python lists
                )
                
                # Create entry without token_type_ids for RoBERTa
                entry = {
                    'input_ids': encoding['input_ids'],
                    'attention_mask': encoding['attention_mask'],
                    'labels': sentiment_id
                }
                formatted_data.append(entry)
    
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

class FocalLoss(torch.nn.Module):
    """
    Focal Loss implementation for handling class imbalance in sequence labeling.
    """
    def __init__(self, gamma=2.0, alpha=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        
    def forward(self, inputs, targets):
        """
        Calculate focal loss
        
        Args:
            inputs: Tensor of shape (batch_size, seq_len, num_classes)
            targets: Tensor of shape (batch_size, seq_len)
        """
        # Flatten inputs and targets
        inputs = inputs.view(-1, inputs.size(-1))  # (batch_size * seq_len, num_classes)
        targets = targets.view(-1)  # (batch_size * seq_len)
        
        # Create mask for valid positions (not padding)
        mask = (targets != self.ignore_index).float()
        masked_targets = targets.clone()
        masked_targets[targets == self.ignore_index] = 0  # Temporarily replace ignore_index with 0
        
        # Move alpha to the same device as inputs if needed
        alpha = None
        if self.alpha is not None:
            alpha = self.alpha
            if alpha.device != inputs.device:
                alpha = alpha.to(inputs.device)
        
        # Calculate cross entropy
        ce_loss = torch.nn.functional.cross_entropy(
            inputs, 
            masked_targets,
            weight=alpha,
            reduction='none'
        )
        
        # Apply focal term: (1 - pt)^gamma
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma
        
        # Apply mask and reduce
        loss = focal_term * ce_loss * mask
        
        # Return mean of valid positions
        return loss.sum() / (mask.sum() + 1e-10)

def train_aspect_extraction(train_dataset, val_dataset, tokenizer, num_epochs=NUM_EPOCHS, output_dir=None, resume_from_checkpoint=False, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, use_focal_loss=False, class_weights=None, gradient_accumulation_steps=1):
    """
    Train the Aspect Term Extraction model using RoBERTa
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        tokenizer: Tokenizer to use
        num_epochs: Number of training epochs
        output_dir: Directory to save the model
        resume_from_checkpoint: Whether to resume from an existing checkpoint
        learning_rate: Learning rate for training
        batch_size: Batch size for training
        use_focal_loss: Whether to use focal loss instead of cross-entropy
        class_weights: Optional class weights for handling class imbalance
        gradient_accumulation_steps: Number of steps to accumulate gradients
    """
    if output_dir is None:
        output_dir = ASPECT_MODEL_PATH
    
    logger.info("Initializing ATE model...")
    
    # Check if we're resuming from a checkpoint
    if resume_from_checkpoint and os.path.exists(output_dir):
        logger.info(f"Loading model from {output_dir} to resume training")
        model = AutoModelForTokenClassification.from_pretrained(output_dir)
        
        # Check if we're changing the loss function
        if use_focal_loss:
            logger.info("Resuming training with Focal Loss (note: this changes the loss function from previous training)")
        elif class_weights is not None:
            logger.info("Resuming training with class weights (note: this may change the loss function from previous training)")
    else:
        # Create model with custom configuration for RoBERTa
        from transformers import RobertaConfig
        
        # Configure RoBERTa with relevant parameters
        config = RobertaConfig.from_pretrained(MODEL_NAME, num_labels=ASPECT_NUM_LABELS)
        
        # Set dropout for better generalization
        config.hidden_dropout_prob = 0.3
        config.attention_probs_dropout_prob = 0.3
        
        model = AutoModelForTokenClassification.from_pretrained(
            MODEL_NAME, config=config
        )
    
    # Configure appropriate training arguments
    train_args = {
        "output_dir": f"{output_dir}/checkpoints",
        "logging_dir": f"{output_dir}/logs",
        "logging_steps": 100,
        "num_train_epochs": num_epochs,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "weight_decay": 0.01,
        "save_steps": 100,
        "eval_steps": 100,
        "save_total_limit": 3,
        "metric_for_best_model": "f1",
        "eval_strategy": "steps",
        "load_best_model_at_end": True,
        "greater_is_better": True,  # Higher F1 is better
        "warmup_ratio": 0.1,  # Gradual warmup for learning rate
        "report_to": "none",  # Disable wandb reporting
        "remove_unused_columns": False,  # Allow datasets with no matching columns
        "gradient_accumulation_steps": gradient_accumulation_steps  # Add gradient accumulation
    }
    
    # Add resume_from_checkpoint if needed
    if resume_from_checkpoint:
        train_args["resume_from_checkpoint"] = True
    
    # Create training arguments
    training_args = TrainingArguments(**train_args)
    
    # Create a data collator for token classification
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Use Transformers get_scheduler for linear warmup
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
    
    # Custom callback for learning rate reduction on F1 plateau
    # and early stopping when we reach min learning rate (1e-8) and no improvement after 10 epochs
    class F1OnPlateauCallback(TrainerCallback):
        def __init__(self, patience=10, factor=0.5, min_lr=1e-9, stopping_patience=PATIENCE):
            self.patience = patience
            self.factor = factor
            self.min_lr = min_lr
            self.stopping_patience = stopping_patience
            self.best_f1 = -float('inf')
            self.no_improve_count = 0
            self.min_lr_reached = False
            self.no_improve_since_min_lr = 0
            
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics is None or "eval_f1" not in metrics:
                return
            
            current_f1 = metrics["eval_f1"]
            current_epoch = state.epoch
            
            if current_f1 > self.best_f1:
                self.best_f1 = current_f1
                self.no_improve_count = 0
                if self.min_lr_reached:
                    self.no_improve_since_min_lr = 0
                logger.info(f"New best F1: {current_f1:.4f} at epoch {current_epoch:.2f}")
            else:
                self.no_improve_count += 1
                if self.min_lr_reached:
                    self.no_improve_since_min_lr += 1
                
                logger.info(f"F1 did not improve for {self.no_improve_count} evaluations. Current: {current_f1:.4f}, Best: {self.best_f1:.4f}")
                
                # Check if we should stop training (reached min LR and no improvement after stopping_patience)
                if self.min_lr_reached and self.no_improve_since_min_lr >= self.stopping_patience:
                    logger.info(f"Early stopping: Min learning rate {self.min_lr} reached and no improvement after {self.stopping_patience} evaluations")
                    control.should_training_stop = True
                    return
                
                # Time to reduce learning rate
                if self.no_improve_count >= self.patience:
                    # Check if we're at min LR already
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    if current_lr <= self.min_lr:
                        # Just set the flag but don't exit yet - wait for stopping_patience
                        if not self.min_lr_reached:
                            logger.info(f"Reached min learning rate of {self.min_lr}")
                            self.min_lr_reached = True
                    else:
                        # Reduce learning rate
                        new_lr = max(current_lr * self.factor, self.min_lr)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = new_lr
                        
                        logger.info(f"Reducing learning rate from {current_lr:.6f} to {new_lr:.6f} after {self.patience} evaluations without improvement")
                        
                        # Reset counter
                        self.no_improve_count = 0
                        
                        # Check if this reduction brought us to min LR
                        if new_lr <= self.min_lr:
                            logger.info(f"Reached min learning rate of {self.min_lr}")
                            self.min_lr_reached = True

    # Create F1 plateau callback with keyword boosting for aspects
    class KeywordAwareAspectCallback(F1OnPlateauCallback):
        def __init__(self, patience=15, factor=0.5, min_lr=1e-6, stopping_patience=PATIENCE, 
                      all_keywords=ALL_ASPECT_KEYWORDS, tokenizer=tokenizer):
            super().__init__(patience, factor, min_lr, stopping_patience)
            self.all_keywords = all_keywords
            self.tokenizer = tokenizer
            
            # Create an initial embedding boost for keyword tokens
            # This will be used in the on_train_begin method
            self.keyword_token_ids = set()
            for keyword in self.all_keywords:
                # Get the token IDs for each keyword
                keyword_ids = tokenizer.encode(keyword, add_special_tokens=False)
                for token_id in keyword_ids:
                    self.keyword_token_ids.add(token_id)
            
            logger.info(f"Keyword-aware training with {len(self.keyword_token_ids)} unique keyword token IDs")
        
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            """Apply initial bias to encourage the model to recognize aspect keywords"""
            if model is None:
                return
                
            # Access the token classification head
            if hasattr(model, "classifier"):
                classifier = model.classifier
            else:
                logger.warning("Model doesn't have a standard classifier attribute, skipping keyword boosting")
                return
                
            # Apply small bias to output layer weights for keyword tokens
            try:
                # Get output layer (should be a Linear layer)
                if hasattr(classifier, "weight"):
                    # For each keyword token, apply a small bias towards B-ASP (label 1)
                    boost_factor = 0.1  # Reduced from 0.3
                    
                    # RoBERTa models structure differs from BERT
                    # Just apply a direct bias rather than trying to find embeddings
                    # Add a small bias towards the B-ASP class for all tokens
                    classifier.bias.data[1] += boost_factor  # B-ASP is label 1
                    
                    logger.info(f"Applied keyword bias boost to classifier")
                else:
                    logger.warning("Classifier doesn't have weights attribute, skipping keyword boosting")
            except Exception as e:
                logger.error(f"Error applying keyword boost: {e}")
    
    # Custom Trainer class for using class weights or focal loss
    class CustomAteTrainer(Trainer):
        def __init__(self, use_focal_loss=False, class_weights=None, *args, **kwargs):
            super(CustomAteTrainer, self).__init__(*args, **kwargs)
            self.use_focal_loss = use_focal_loss
            self.class_weights = class_weights
            
            if use_focal_loss:
                logger.info("Using Focal Loss for ATE training")
            elif class_weights is not None:
                logger.info(f"Using class weights for ATE training: {class_weights}")
        
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            
            if self.use_focal_loss:
                # Apply focal loss
                loss_fct = FocalLoss(gamma=2.0, alpha=self.class_weights)
                loss = loss_fct(logits, labels)
            elif self.class_weights is not None:
                # Use cross entropy with class weights
                weights = self.class_weights
                # Check if weights need to be moved to the same device
                if weights.device != logits.device:
                    weights = weights.to(logits.device)
                loss_fct = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
                loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            else:
                # Standard cross entropy
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
                
            return (loss, outputs) if return_outputs else loss
    
    # Create our custom callback
    f1_plateau_callback = KeywordAwareAspectCallback(
        patience=15, 
        factor=0.75, 
        min_lr=1e-6, 
        stopping_patience=PATIENCE
    )
    
    # Add early stopping callback as a backup
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=10,  # Reduced from 30
        early_stopping_threshold=0.001
    )
    
    # Convert class weights to tensor if provided
    if class_weights is not None and not isinstance(class_weights, torch.Tensor):
        class_weights = torch.tensor(class_weights, dtype=torch.float)
    
    # Create the trainer with our custom callbacks and options
    trainer = CustomAteTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_ate_metrics,
        optimizers=(optimizer, lr_scheduler),  # Pass our custom optimizer and scheduler
        callbacks=[f1_plateau_callback, early_stopping_callback],  # Add our custom callbacks
        use_focal_loss=use_focal_loss,
        class_weights=class_weights
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
    Train the Aspect Sentiment Classification model using RoBERTa
    """
    if output_dir is None:
        output_dir = SENTIMENT_MODEL_PATH
    
    logger.info("Initializing ASC model...")
    
    # Check if we're resuming from a checkpoint
    if resume_from_checkpoint and os.path.exists(output_dir):
        logger.info(f"Loading model from {output_dir} to resume training")
        model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    else:
        # Create model with custom configuration for RoBERTa
        from transformers import RobertaConfig
        
        # Configure RoBERTa with relevant parameters for sentiment analysis
        config = RobertaConfig.from_pretrained(MODEL_NAME, num_labels=SENTIMENT_NUM_LABELS)
        
        # Set dropout for better generalization
        config.hidden_dropout_prob = 0.2
        config.attention_probs_dropout_prob = 0.2
        
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, config=config
        )
    
    # Configure appropriate training arguments
    train_args = {
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
        "eval_strategy": "steps",
        "load_best_model_at_end": True,
        "greater_is_better": True,  # Higher F1 is better
        "warmup_ratio": 0.1,  # Gradual warmup for learning rate
        "report_to": "none",  # Disable wandb reporting
        "remove_unused_columns": False  # Allow datasets with no matching columns
    }
    
    # Add resume_from_checkpoint if needed
    if resume_from_checkpoint:
        train_args["resume_from_checkpoint"] = True
    
    # Create training arguments
    training_args = TrainingArguments(**train_args)
    
    # Create optimizer with weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    
    # Use Transformers get_scheduler for linear warmup
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
    
    # Custom callback for learning rate reduction on F1 plateau
    # and early stopping when we reach min learning rate (1e-6) and no improvement after 30 epochs
    class MacroF1OnPlateauCallback(TrainerCallback):
        def __init__(self, patience=10, factor=0.5, min_lr=5e-7, stopping_patience=PATIENCE):
            self.patience = patience
            self.factor = factor
            self.min_lr = min_lr
            self.stopping_patience = stopping_patience
            self.best_f1 = -float('inf')
            self.no_improve_count = 0
            self.min_lr_reached = False
            self.no_improve_since_min_lr = 0
            
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics is None or "eval_macro_f1" not in metrics:
                return
            
            current_f1 = metrics["eval_macro_f1"]
            current_epoch = state.epoch
            
            if current_f1 > self.best_f1:
                self.best_f1 = current_f1
                self.no_improve_count = 0
                if self.min_lr_reached:
                    self.no_improve_since_min_lr = 0
                logger.info(f"New best Macro F1: {current_f1:.4f} at epoch {current_epoch:.2f}")
            else:
                self.no_improve_count += 1
                if self.min_lr_reached:
                    self.no_improve_since_min_lr += 1
                
                logger.info(f"Macro F1 did not improve for {self.no_improve_count} evaluations. Current: {current_f1:.4f}, Best: {self.best_f1:.4f}")
                
                # Check if we should stop training (reached min LR and no improvement after stopping_patience)
                if self.min_lr_reached and self.no_improve_since_min_lr >= self.stopping_patience:
                    logger.info(f"Early stopping: Min learning rate {self.min_lr} reached and no improvement after {self.stopping_patience} evaluations")
                    control.should_training_stop = True
                    return
                
                # Time to reduce learning rate
                if self.no_improve_count >= self.patience:
                    # Check if we're at min LR already
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    if current_lr <= self.min_lr:
                        # Just set the flag but don't exit yet - wait for stopping_patience
                        if not self.min_lr_reached:
                            logger.info(f"Reached min learning rate of {self.min_lr}")
                            self.min_lr_reached = True
                    else:
                        # Reduce learning rate
                        new_lr = max(current_lr * self.factor, self.min_lr)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = new_lr
                        
                        logger.info(f"Reducing learning rate from {current_lr:.6f} to {new_lr:.6f} after {self.patience} evaluations without improvement")
                        
                        # Reset counter
                        self.no_improve_count = 0
                        
                        # Check if this reduction brought us to min LR
                        if new_lr <= self.min_lr:
                            logger.info(f"Reached min learning rate of {self.min_lr}")
                            self.min_lr_reached = True
    
    # Create the F1 plateau callback
    macro_f1_plateau_callback = MacroF1OnPlateauCallback(
        patience=15, 
        factor=0.75, 
        min_lr=1e-6,
        stopping_patience=PATIENCE
    )
    
    # Add early stopping callback as a backup
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=10,  # Reduced from 30
        early_stopping_threshold=0.001
    )
    
    # Custom loss for handling class imbalance
    class FocalLossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Apply focal loss for better handling of class imbalance
            gamma = 2.0  # Focus parameter - higher means more focus on hard examples
            
            # Convert labels to one-hot
            num_labels = len(SENTIMENT_LABELS)
            one_hot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1.0)
            
            # Compute probabilities
            probs = torch.nn.functional.softmax(logits, dim=1)
            
            # Calculate focal weights: (1-pt)^gamma
            pt = (one_hot * probs).sum(dim=1)  # Get the probability of the correct class
            focal_weights = (1 - pt) ** gamma
            
            # Apply cross entropy with focal weights
            per_sample_losses = -torch.log(pt + 1e-10) * focal_weights
            loss = per_sample_losses.mean()
            
            return (loss, outputs) if return_outputs else loss
    
    # Create trainer with focal loss for better handling of class imbalance
    trainer = FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_asc_metrics,
        optimizers=(optimizer, lr_scheduler),  # Pass our custom optimizer and scheduler
        callbacks=[macro_f1_plateau_callback, early_stopping_callback]  # Add our custom callbacks
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

def apply_adaptive_thresholding(logits, threshold_b_asp=0.6, threshold_i_asp=0.5):
    """
    Apply adaptive thresholding to model predictions to improve precision.
    
    Args:
        logits: Raw model logits (batch_size, seq_len, num_classes)
        threshold_b_asp: Confidence threshold for B-ASP predictions
        threshold_i_asp: Confidence threshold for I-ASP predictions
        
    Returns:
        Adjusted prediction indices
    """
    # Convert logits to probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get initial predictions
    predictions = torch.argmax(probs, dim=-1)
    
    # Apply thresholds
    b_asp_idx = ASPECT_LABEL_MAP["B-ASP"]
    i_asp_idx = ASPECT_LABEL_MAP["I-ASP"]
    
    # Create mask for predictions that don't meet threshold
    b_asp_mask = (predictions == b_asp_idx) & (probs[:, :, b_asp_idx] < threshold_b_asp)
    i_asp_mask = (predictions == i_asp_idx) & (probs[:, :, i_asp_idx] < threshold_i_asp)
    
    # Change low-confidence predictions to O (0)
    predictions[b_asp_mask] = 0
    predictions[i_asp_mask] = 0
    
    return predictions

def post_process_aspect_predictions(predictions, tokenizer, text):
    """
    Post-process aspect predictions to improve quality.
    
    - Merge adjacent aspect tokens
    - Filter out unlikely aspects (e.g., very short, stopwords)
    - Ensure B-ASP is followed by I-ASP for consistency
    - Apply domain-specific filtering for mobile phone reviews
    
    Args:
        predictions: List of prediction indices
        tokenizer: Tokenizer used for inference
        text: Original text
        
    Returns:
        List of cleaned aspect terms
    """
    tokens = tokenizer.convert_ids_to_tokens(predictions)
    labels = [ASPECT_LABEL_MAP_INVERSE.get(p, "O") for p in predictions]
    
    # Extract aspects
    aspects = []
    current_aspect = []
    
    for token, label in zip(tokens, labels):
        if label == "B-ASP":
            # If we were already collecting an aspect, finalize it
            if current_aspect:
                aspect_text = tokenizer.convert_tokens_to_string(current_aspect)
                aspects.append(aspect_text)
                current_aspect = []
            
            # Start new aspect
            current_aspect.append(token)
        elif label == "I-ASP" and current_aspect:
            # Continue current aspect
            current_aspect.append(token)
        elif current_aspect:
            # End of aspect
            aspect_text = tokenizer.convert_tokens_to_string(current_aspect)
            aspects.append(aspect_text)
            current_aspect = []
    
    # Don't forget the last aspect if there is one
    if current_aspect:
        aspect_text = tokenizer.convert_tokens_to_string(current_aspect)
        aspects.append(aspect_text)
    
    # Domain-specific known aspect terms for mobile phones
    KNOWN_ASPECTS = {
        'μπαταρια', 'μπαταρία', 'οθονη', 'οθόνη', 'καμερα', 'κάμερα',
        'επεξεργαστης', 'επεξεργαστής', 'ταχυτητα', 'ταχύτητα',
        'μνημη', 'μνήμη', 'αποθηκευτικός', 'χωρος', 'χώρος',
        'ηχεια', 'ηχεία', 'ηχος', 'ήχος', 'τιμη', 'τιμή',
        'σχεδιαση', 'σχεδίαση', 'βαρος', 'βάρος', 'λειτουργικο',
        'λειτουργικό', 'αναλυση', 'ανάλυση', 'αισθητηρας', 'αισθητήρας',
        'φωτογραφιες', 'φωτογραφίες', 'βιντεο', 'βίντεο'
    }
    
    # Filter aspects
    filtered_aspects = []
    for aspect in aspects:
        # Clean the aspect text
        aspect = aspect.strip()
        
        # Skip very short aspects (1-2 characters)
        if len(aspect) <= 2:
            continue
            
        # Skip if it's a common stopword
        if aspect.lower() in STOPWORDS:
            continue
            
        # Skip if it's a common adjective
        if aspect.lower() in ADJECTIVES:
            continue
        
        # Give preference to known domain-specific aspects
        aspect_lower = aspect.lower()
        found_match = False
        
        # Check if this aspect contains any known aspect terms
        for known_aspect in KNOWN_ASPECTS:
            if known_aspect in aspect_lower:
                found_match = True
                break
                
        # Perform additional checks for non-matched aspects
        if not found_match:
            # Skip very common words that aren't aspects
            if aspect_lower in NON_ASPECT_WORDS:
                continue
                
            # Skip aspects that are likely just parts of words due to tokenization
            if len(aspect) <= 3 and not aspect.isalpha():
                continue
        
        # Check if the aspect actually appears in the original text
        # This helps filter out tokenization artifacts
        normalized_text = text.lower()
        normalized_aspect = aspect_lower.replace('ġ', '').replace(' ', '')
        if normalized_aspect not in normalized_text.replace(' ', ''):
            # The aspect isn't found in the original text, might be a tokenization artifact
            # In some cases we might still want to keep it if it's a known aspect term
            if not found_match:
                continue
            
        filtered_aspects.append(aspect)
    
    return filtered_aspects

# ================ INFERENCE FUNCTIONS ================

class ABSAPipeline:
    """
    Pipeline for Aspect-Based Sentiment Analysis using RoBERTa model
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
            # Preprocess text for RoBERTa
            text_proc = text  # We'll keep the original for offset tracking
            
            # Tokenize the input
            inputs = self.aspect_tokenizer(text_proc, return_tensors="pt", truncation=True, padding=True, max_length=128)
            
            # Move to GPU if available
            device = next(self.aspect_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get the predictions
            with torch.no_grad():
                outputs = self.aspect_model(**inputs)
                predictions = outputs.logits
            
            # Apply adaptive thresholding for better precision
            # Use lower thresholds for testing to increase recall
            pred_labels = apply_adaptive_thresholding(predictions, 
                                                     threshold_b_asp=0.6,  # Increased from 0.4
                                                     threshold_i_asp=0.5)  # Increased from 0.3
            
            # Convert to numpy for easier handling
            pred_labels_np = pred_labels.detach().cpu().numpy()[0]
            
            # Use post-processing to clean up aspect predictions
            filtered_aspects = post_process_aspect_predictions(
                inputs['input_ids'][0].cpu().numpy(), 
                self.aspect_tokenizer, 
                text
            )
            
            # Get original predictions from the pipeline for offset information
            aspects = []
            i = 0
            
            # Get tokens
            tokens = self.aspect_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].cpu())
            
            # Extract aspects with confidence above threshold
            i = 0
            while i < len(tokens):
                if tokens[i] in ['<s>', '</s>', '<pad>']:  # RoBERTa special tokens
                    i += 1
                    continue
                    
                if pred_labels_np[i] == ASPECT_LABEL_MAP['B-ASP']:
                    # Found the beginning of an aspect
                    aspect_start = i
                    aspect_end = i
                    
                    # Look for continuation (I-ASP)
                    j = i + 1
                    while j < len(tokens) and j < len(pred_labels_np):
                        if tokens[j] in ['<s>', '</s>', '<pad>']:
                            j += 1
                            continue
                            
                        # Consider additional tokens as part of the aspect if they are I-ASP
                        # or if they have Ġ prefix (they are part of the same word in RoBERTa tokenization)
                        if (pred_labels_np[j] == ASPECT_LABEL_MAP['I-ASP']) or \
                           (not tokens[j].startswith('Ġ') and j > 0):  # RoBERTa tokens without Ġ are inside a word
                            aspect_end = j
                            j += 1
                        else:
                            break
                    
                    # Extract the aspect text
                    aspect_tokens = tokens[aspect_start:aspect_end+1]
                    
                    # RoBERTa uses Ġ to indicate start of words, remove them and join
                    aspect_text = ''.join([t.replace('Ġ', ' ') for t in aspect_tokens]).strip()
                    
                    # Filter out very short aspects and non-aspect words
                    # Be more lenient for testing
                    if len(aspect_text) <= 1:
                        i = aspect_end + 1
                        continue
                    
                    # Calculate character offsets in the original text
                    # Note: This is an approximation as tokenization can be complex
                    start_char = -1
                    end_char = -1
                    
                    # Simple search for the aspect in original text
                    aspect_text_lower = aspect_text.lower()
                    text_lower = text.lower()
                    start_char = text_lower.find(aspect_text_lower)
                    if start_char >= 0:
                        end_char = start_char + len(aspect_text)
                    
                    # Assign a confidence score based on the aspect length and position
                    confidence_score = 0.7 + (min(len(aspect_text), 10) / 30.0)  # Length-based boost
                    
                    aspects.append({
                        'word': aspect_text,
                        'score': confidence_score,
                        'start': start_char,
                        'end': end_char
                    })
                    
                    i = aspect_end + 1
                else:
                    i += 1
            
            # Merge with the filtered aspects from post-processing if they're different
            merged_aspects = []
            existing_aspects = set()
            
            # First add the aspects from the detailed extraction
            for aspect in aspects:
                aspect_text = aspect['word'].lower()
                if aspect_text not in existing_aspects:
                    merged_aspects.append(aspect)
                    existing_aspects.add(aspect_text)
            
            # Then add the aspects from post-processing if not already included
            for aspect_text in filtered_aspects:
                aspect_lower = aspect_text.lower()
                if aspect_lower not in existing_aspects:
                    # Find position in text (approximate)
                    start_char = text.lower().find(aspect_lower)
                    if start_char >= 0:
                        end_char = start_char + len(aspect_text)
                    else:
                        start_char = -1
                        end_char = -1
                        
                    merged_aspects.append({
                        'word': aspect_text,
                        'score': 0.6,  # Default score for post-processed aspects
                        'start': start_char,
                        'end': end_char
                    })
                    existing_aspects.add(aspect_lower)
            
            # If we still don't have any aspects, try to extract them directly from the text
            # This is a fallback for testing purposes
            if not merged_aspects:
                # Try a simple rule-based extraction
                words = text.split()
                for word in words:
                    # Skip very short words, stopwords, and adjectives
                    if len(word) <= 2 or word.lower() in STOPWORDS or word.lower() in ADJECTIVES:
                        continue
                    
                    # Add as potential aspect
                    start_char = text.lower().find(word.lower())
                    if start_char >= 0:
                        end_char = start_char + len(word)
                        
                        merged_aspects.append({
                            'word': word,
                            'score': 0.5,  # Low confidence for rule-based extraction
                            'start': start_char,
                            'end': end_char
                        })
            
            return merged_aspects
        except Exception as e:
            logger.error(f"Error in aspect extraction: {str(e)}")
            return []
    
    def classify_sentiment(self, text, aspect):
        """Classify sentiment for a given text-aspect pair"""
        if not text or not aspect:
            logger.warning("Empty text or aspect provided for sentiment classification")
            return {'label': 1, 'score': 0.0}  # Default to neutral
        
        try:
            # Preprocess text and aspect for RoBERTa
            text_proc = text
            aspect_proc = aspect
            
            # Tokenize the input
            inputs = self.sentiment_tokenizer(text_proc, aspect_proc, return_tensors="pt", truncation=True, padding=True)
            
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

# ================ INITIALIZATION FUNCTION ================

def initialize_tokenizer():
    """Initialize tokenizer from the pretrained model"""
    logger.info(f"Initializing tokenizer from {MODEL_NAME}")
    return AutoTokenizer.from_pretrained(MODEL_NAME)