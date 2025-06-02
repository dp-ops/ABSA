import json
import time
import os
import torch
import numpy as np
import logging
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
from datasets import Dataset
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from sklearn.metrics import  accuracy_score
import torch.nn as nn

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "nlpaueb/bert-base-greek-uncased-v1"
SAVED_MODELS_DIR = "models/saved_models_final"
NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_LENGTH = 256

# Create saved_models directory if it doesn't exist
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# Aspect categories mapping
ASPECT_CATEGORIES = [
    "Ποιότητα κλήσης", "Φωτογραφίες", "Καταγραφή Video", "Ταχύτητα",
    "Ανάλυση οθόνης", "Μπαταρία", "Σχέση ποιότητας τιμής", "Μουσική"
]
ASPECT_TO_ID = {aspect: idx for idx, aspect in enumerate(ASPECT_CATEGORIES)}
ID_TO_ASPECT = {idx: aspect for aspect, idx in ASPECT_TO_ID.items()}

# Sentiment labels
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
SENTIMENT_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_SENTIMENT = {0: "negative", 1: "neutral", 2: "positive"}

# Model paths
ATC_MODEL_PATH = f"{SAVED_MODELS_DIR}/aspect_classification_model"
ASC_MODEL_PATH = f"{SAVED_MODELS_DIR}/aspect_sentiment_model"

# ================ MULTI-LABEL BERT MODEL FOR ASPECT CLASSIFICATION ================

class MultiLabelBertForAspectClassification(nn.Module):
    """
    Multi-label BERT model for aspect classification.
    Takes text and predicts which aspects are present (multi-label classification).
    """
    def __init__(self, model_name=MODEL_NAME, num_labels=len(ASPECT_CATEGORIES)):
        super().__init__()
        self.num_labels = num_labels
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        return outputs

# ================ DATASET LOADING FUNCTIONS ================

def load_atc_dataset(file_path, tokenizer):
    """
    Load and preprocess the ATC (Aspect Term Classification) dataset.
    Multi-label classification: predict which aspects are present in the text.
    """
    logger.info(f"Loading aspect classification dataset from {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    formatted_data = []
    for item in data:
        text = item['text_processed']  # Use preprocessed text
        
        # Create multi-label target
        # Initialize all aspects as 0.0 (not present) - use float for multi-label classification
        aspect_labels = [0.0] * len(ASPECT_CATEGORIES)
        
        # Set to 1.0 for aspects that are present
        for aspect_info in item.get('aspects_present', []):
            aspect_name = aspect_info['aspect_category']
            if aspect_name in ASPECT_TO_ID:
                aspect_labels[ASPECT_TO_ID[aspect_name]] = 1.0
        
        # Tokenize the text
        encoding = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors=None
        )
        
        formatted_data.append({
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'token_type_ids': encoding.get('token_type_ids', [0] * len(encoding['input_ids'])),
            'labels': aspect_labels  # Multi-label: list of 0s and 1s
        })

    logger.info(f"Loaded {len(formatted_data)} samples for ATC")
    return Dataset.from_list(formatted_data)

def load_asc_dataset(file_path, tokenizer):
    """
    Load and preprocess the ASC (Aspect Sentiment Classification) dataset.
    For each aspect-text pair, predict sentiment.
    """
    logger.info(f"Loading aspect sentiment classification dataset from {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    formatted_data = []
    for item in data:
        text = item['text_processed']
        
        # Create entries for each aspect and its sentiment
        for aspect_info in item.get('aspects_present', []):
            aspect_name = aspect_info['aspect_category']
            sentiment_id = aspect_info['sentiment_id']
            
            # Encode text + aspect as input
            encoding = tokenizer(
                text,
                aspect_name,
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors=None
            )
            
            formatted_data.append({
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'token_type_ids': encoding.get('token_type_ids', [0] * len(encoding['input_ids'])),
                'labels': sentiment_id
            })

    logger.info(f"Loaded {len(formatted_data)} samples for ASC")
    return Dataset.from_list(formatted_data)

# ================ METRIC COMPUTATION FUNCTIONS ================

def compute_atc_metrics(p):
    """
    Compute metrics for Aspect Term Classification (multi-label).
    """
    predictions, labels = p
    
    # Apply sigmoid to get probabilities, then threshold at 0.5
    predictions = torch.sigmoid(torch.tensor(predictions))
    predictions = (predictions > 0.5).int().numpy()
    
    # Convert to numpy if needed
    labels = np.array(labels)
    predictions = np.array(predictions)
    
    # Calculate metrics for each aspect
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    for i in range(len(ASPECT_CATEGORIES)):
        y_true = labels[:, i]
        y_pred = predictions[:, i]
        
        # Skip if no positive examples
        if y_true.sum() == 0:
            continue
            
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision, recall, _, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
    
    # Calculate macro averages
    macro_f1 = np.mean(f1_scores) if f1_scores else 0.0
    macro_precision = np.mean(precision_scores) if precision_scores else 0.0
    macro_recall = np.mean(recall_scores) if recall_scores else 0.0
    
    # Calculate subset accuracy (exact match)
    subset_accuracy = accuracy_score(labels, predictions)
    
    return {
        "f1": macro_f1,
        "precision": macro_precision,
        "recall": macro_recall,
        "subset_accuracy": subset_accuracy,
    }

def compute_asc_metrics(p):
    """
    Compute metrics for Aspect Sentiment Classification.
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    
    try:
        results = classification_report(
            labels,
            predictions,
            target_names=SENTIMENT_LABELS,
            output_dict=True,
            digits=4,
            zero_division=0
        )
        
        return {
            "accuracy": results["accuracy"],
            "macro_precision": results["macro avg"]["precision"],
            "macro_recall": results["macro avg"]["recall"],
            "macro_f1": results["macro avg"]["f1-score"],
            "neg_f1": results.get("negative", {}).get("f1-score", 0.0),
            "neu_f1": results.get("neutral", {}).get("f1-score", 0.0),
            "pos_f1": results.get("positive", {}).get("f1-score", 0.0)
        }
    except Exception as e:
        logger.warning(f"Error computing ASC metrics: {e}")
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "neg_f1": 0.0,
            "neu_f1": 0.0,
            "pos_f1": 0.0
        }

# ================ CUSTOM TRAINER FOR WEIGHTED LOSS (ASC) ================
class WeightedTrainerASC(Trainer):
    def __init__(self, *args, class_weights=None, use_focal_loss=True, focal_alpha=None, focal_gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        
        if class_weights is not None:
            # Ensure class_weights are on the correct device, which is model's device
            self.class_weights = torch.tensor(class_weights, device=self.model.device, dtype=torch.float)
            # Use class weights as focal loss alpha if not provided separately
            self.focal_alpha = focal_alpha if focal_alpha is not None else self.class_weights
        else:
            self.class_weights = None
            self.focal_alpha = focal_alpha

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs) # model is already on the correct device
        logits = outputs.get("logits")
        
        if self.use_focal_loss and self.focal_alpha is not None:
            # Focal Loss implementation for highly imbalanced data
            ce_loss = nn.functional.cross_entropy(logits, labels, reduction='none')
            pt = torch.exp(-ce_loss)
            
            # Get alpha for each sample based on its true label
            alpha_t = self.focal_alpha[labels]
            focal_loss = alpha_t * (1 - pt) ** self.focal_gamma * ce_loss
            loss = focal_loss.mean()
            
        elif self.class_weights is not None:
            # Standard weighted cross entropy
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            # Standard cross entropy
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            
        return (loss, outputs) if return_outputs else loss

# ================ MODEL TRAINING FUNCTIONS ================

def train_aspect_classification(train_dataset, val_dataset, tokenizer, num_epochs=NUM_EPOCHS, 
                               output_dir=None, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE,
                               gradient_clipping_norm=1.0, resume_from_checkpoint=False):
    """
    Train the Aspect Term Classification model (multi-label).
    """
    if output_dir is None:
        output_dir = ATC_MODEL_PATH
    
    logger.info("Initializing ATC model...")
    
    model = MultiLabelBertForAspectClassification(
        model_name=MODEL_NAME,
        num_labels=len(ASPECT_CATEGORIES)
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/checkpoints",
        logging_dir=f"{output_dir}/logs",
        logging_steps=25,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_steps=50,
        eval_steps=50,
        save_total_limit=3,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        warmup_ratio=0.1,
        report_to="none",
        max_grad_norm=gradient_clipping_norm,
        lr_scheduler_type="reduce_lr_on_plateau",
        lr_scheduler_kwargs={'mode': 'max', 'factor': 0.7, 'patience': 7, 'threshold': 0.0001, 'min_lr': 1e-7}
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_atc_metrics
    )
    
    logger.info("Training ATC model...")
    start_time = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    end_time = time.time()
    
    training_time = end_time - start_time
    logger.info(f"ATC training completed in {training_time:.2f} seconds")
    
    # Save the model
    model.bert.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    metrics['training_time'] = training_time
    
    with open(f"{output_dir}/training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Evaluate the model
    logger.info("Evaluating ATC model...")
    eval_metrics = trainer.evaluate()
    
    with open(f"{output_dir}/evaluation_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=4)
    
    logger.info(f"ATC evaluation metrics: {eval_metrics}")
    return model, tokenizer, eval_metrics

def train_aspect_sentiment(train_dataset, val_dataset, tokenizer, num_epochs=NUM_EPOCHS,
                          output_dir=None, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE,
                          gradient_clipping_norm=1.0, resume_from_checkpoint=False,
                          asc_class_weights=None):
    """
    Train the Aspect Sentiment Classification model.
    """
    if output_dir is None:
        output_dir = ASC_MODEL_PATH
    
    logger.info("Initializing ASC model...")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(SENTIMENT_LABELS)
    )
    
    # Training arguments optimized for highly imbalanced data
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/checkpoints",
        logging_dir=f"{output_dir}/logs",
        logging_steps=25,  # More frequent logging for imbalanced data
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_steps=50,  # More frequent saves
        eval_steps=50,  # More frequent evaluation
        save_total_limit=3,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        warmup_ratio=0.15,  # Increased warmup for stability
        report_to="none",
        #max_grad_norm=gradient_clipping_norm,
        dataloader_drop_last=False,  # Don't drop last batch to preserve minority samples
        lr_scheduler_type="reduce_lr_on_plateau",
        lr_scheduler_kwargs={'mode': 'max', 'factor': 0.7, 'patience': 2, 'threshold': 0.001, 'min_lr': 1e-8}
    )
    
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
    
    # Custom callback for learning rate reduction on plateau (optimized for imbalanced data)
    class MacroF1OnPlateauCallback(TrainerCallback):
        def __init__(self, patience=8, factor=0.6, min_lr=1e-8):  # More aggressive for imbalanced data
            self.patience = patience
            self.factor = factor
            self.min_lr = min_lr
            self.best_f1 = -float('inf')
            self.no_improve_count = 0
        
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics is None or "eval_macro_f1" not in metrics:
                return
            
            current_f1 = metrics["eval_macro_f1"]
            
            if current_f1 > self.best_f1 + 0.001:  # Small threshold to avoid noise
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
    macro_f1_plateau_callback = MacroF1OnPlateauCallback(patience=10, factor=0.7, min_lr=1e-8)
    
    # Use WeightedTrainerASC if class weights are provided
    if asc_class_weights:
        logger.info(f"Using WeightedTrainerASC with class weights: {asc_class_weights}")
        trainer = WeightedTrainerASC(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_asc_metrics,
            class_weights=asc_class_weights,
            optimizers=(optimizer, lr_scheduler),  # Pass our custom optimizer and scheduler
            callbacks=[macro_f1_plateau_callback]  # Add our custom callback
        )
    else:
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
    
    # Save the model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    metrics['training_time'] = training_time
    
    with open(f"{output_dir}/training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Evaluate the model
    logger.info("Evaluating ASC model...")
    eval_metrics = trainer.evaluate()
    
    with open(f"{output_dir}/evaluation_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=4)
    
    logger.info(f"ASC evaluation metrics: {eval_metrics}")
    return model, tokenizer, eval_metrics

# ================ INFERENCE PIPELINE ================

class ABSAPipeline:
    """
    Complete ABSA Pipeline: Aspect Classification + Sentiment Classification
    """
    def __init__(self, atc_model_path=ATC_MODEL_PATH, asc_model_path=ASC_MODEL_PATH):
        logger.info(f"Loading models from {atc_model_path} and {asc_model_path}")
        
        # Load ATC model (Aspect Classification)
        self.atc_tokenizer = AutoTokenizer.from_pretrained(atc_model_path)
        self.atc_model = AutoModelForSequenceClassification.from_pretrained(atc_model_path)
        self.atc_model.eval()
        
        # Load ASC model (Sentiment Classification)
        self.asc_tokenizer = AutoTokenizer.from_pretrained(asc_model_path)
        self.asc_model = AutoModelForSequenceClassification.from_pretrained(asc_model_path)
        self.asc_model.eval()
        
        logger.info("ABSA Pipeline initialized successfully")
    
    def classify_aspects(self, text, threshold=0.5):
        """
        Classify which aspects are present in the text.
        """
        # Tokenize
        encoding = self.atc_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        
        # Predict
        with torch.no_grad():
            outputs = self.atc_model(**encoding)
            predictions = torch.sigmoid(outputs.logits)
        
        # Get aspects above threshold
        detected_aspects = []
        for i, score in enumerate(predictions[0]):
            if score > threshold:
                detected_aspects.append({
                    'aspect': ASPECT_CATEGORIES[i],
                    'confidence': float(score)
                })
        
        return detected_aspects
    
    def classify_sentiment(self, text, aspect):
        """
        Classify sentiment for a given text-aspect pair.
        """
        # Tokenize text + aspect
        encoding = self.asc_tokenizer(
            text,
            aspect,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        
        # Predict
        with torch.no_grad():
            outputs = self.asc_model(**encoding)
            predictions = torch.softmax(outputs.logits, dim=1)
        
        # Get predicted sentiment
        predicted_id = torch.argmax(predictions, dim=1).item()
        confidence = float(predictions[0][predicted_id])
        
        return {
            'sentiment': ID_TO_SENTIMENT[predicted_id],
            'confidence': confidence,
            'sentiment_id': predicted_id
        }
    
    def analyze(self, text, aspect_threshold=0.5):
        """
        Complete ABSA analysis: detect aspects and classify their sentiment.
        """
        logger.info(f"Analyzing text: {text}")
        
        # Step 1: Detect aspects
        detected_aspects = self.classify_aspects(text, aspect_threshold)
        
        if not detected_aspects:
            logger.info("No aspects detected")
            return []
        
        logger.info(f"Detected {len(detected_aspects)} aspects: {[a['aspect'] for a in detected_aspects]}")
        
        # Step 2: Classify sentiment for each detected aspect
        results = []
        for aspect_info in detected_aspects:
            aspect_name = aspect_info['aspect']
            aspect_confidence = aspect_info['confidence']
            
            sentiment_result = self.classify_sentiment(text, aspect_name)
            
            results.append({
                'aspect': aspect_name,
                'aspect_confidence': aspect_confidence,
                'sentiment': sentiment_result['sentiment'],
                'sentiment_confidence': sentiment_result['confidence'],
                'sentiment_id': sentiment_result['sentiment_id']
            })
        
        return results

# ================ UTILITY FUNCTIONS ================

def initialize_tokenizer():
    """Initialize tokenizer from the pretrained model"""
    logger.info(f"Initializing tokenizer from {MODEL_NAME}")
    return AutoTokenizer.from_pretrained(MODEL_NAME)

def evaluate_pipeline_on_test_data(pipeline, test_file_path, num_samples=None):
    """
    Evaluate the complete pipeline on test data.
    """
    logger.info(f"Evaluating pipeline on {test_file_path}")
    
    # Load test data
    test_data = []
    with open(test_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    
    if num_samples:
        test_data = test_data[:num_samples]
    
    # Evaluate
    correct_aspects = 0
    total_aspects = 0
    correct_sentiments = 0
    total_sentiments = 0
    
    for item in test_data:
        text = item['text_processed']
        gold_aspects = {asp['aspect_category']: asp['sentiment_id'] 
                       for asp in item.get('aspects_present', [])}
        
        # Get predictions
        predictions = pipeline.analyze(text)
        pred_aspects = {pred['aspect']: pred['sentiment_id'] 
                       for pred in predictions}
        
        # Count correct aspect detections
        for aspect in gold_aspects:
            total_aspects += 1
            if aspect in pred_aspects:
                correct_aspects += 1
                
                # Count correct sentiment predictions
                total_sentiments += 1
                if pred_aspects[aspect] == gold_aspects[aspect]:
                    correct_sentiments += 1
    
    # Calculate metrics
    aspect_accuracy = correct_aspects / total_aspects if total_aspects > 0 else 0
    sentiment_accuracy = correct_sentiments / total_sentiments if total_sentiments > 0 else 0
    
    logger.info(f"Aspect Detection Accuracy: {aspect_accuracy:.4f}")
    logger.info(f"Sentiment Classification Accuracy: {sentiment_accuracy:.4f}")
    
    return {
        'aspect_accuracy': aspect_accuracy,
        'sentiment_accuracy': sentiment_accuracy,
        'total_aspects': total_aspects,
        'correct_aspects': correct_aspects,
        'total_sentiments': total_sentiments,
        'correct_sentiments': correct_sentiments
    }
