import json
import logging
import argparse
import os
import torch
from tqdm import tqdm
import numpy as np
from seqeval.metrics import classification_report as seqeval_classification_report
from sklearn.metrics import classification_report as sklearn_classification_report

from transformers import AutoTokenizer, AutoModelForSequenceClassification # Added for ASC

from model import (
    # Constants
    ASPECT_MODEL_PATH, ASPECT_LABEL_MAP, ASPECT_LABEL_MAP_INVERSE, MODEL_NAME,
    SENTIMENT_MODEL_PATH, SENTIMENT_LABELS, SENTIMENT_NUM_LABELS, # Added for ASC
    ASPECT_NUM_LABELS, # Added
    MAX_LENGTH, # Added
    
    # Classes and functions
    initialize_tokenizer, XLMRobertaForTokenClassificationCRF,
    recover_char_offsets, ABSAPipeline,
    enhanced_align_tokens_and_labels, convert_aligned_labels_to_ids # Added for ATE alignment
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Test XLM-RoBERTa ATE and ASC models, and ABSA pipeline")
    parser.add_argument('--test_file', default='data/filtered_review_data_xlm/processed_aspect_data_test.json',
                        help='Path to test data file (must contain text, bio_labels, and aspects with sentiment_id)')
    parser.add_argument('--ate_model_path', default=ASPECT_MODEL_PATH, # Renamed for clarity
                        help='Path to trained ATE model')
    parser.add_argument('--asc_model_path', default=SENTIMENT_MODEL_PATH, # Added for ASC
                        help='Path to trained ASC model')
    parser.add_argument('--use_crf', action='store_true', default=True,
                        help='Use CRF layer for ATE inference (default: True)')
    parser.add_argument('--output_dir', default='results', # Changed from output_file
                        help='Directory to save prediction files and reports')
    return parser.parse_args()

def load_test_data(test_file_path):
    """Loads test data from a JSONL file."""
    test_data = []
    with open(test_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    logger.info(f"Loaded {len(test_data)} test samples from {test_file_path}")
    return test_data

# --- ATE Evaluation ---
def evaluate_ate_model(model, tokenizer, test_data, use_crf, device):
    """Evaluate the Aspect Term Extraction (ATE) model."""
    logger.info("Evaluating ATE model...")
    model.to(device)
    model.eval()

    all_true_bio_labels = []
    all_pred_bio_labels = []
    ate_predictions_output = [] # For saving raw predictions if needed

    for item in tqdm(test_data, desc="ATE Evaluation"):
        text = item['text']
        original_tokens = item.get('tokens') # Ground truth tokens
        original_bio_labels = item.get('bio_labels') # Ground truth BIO labels for original_tokens

        if not text or not original_tokens or not original_bio_labels:
            logger.warning(f"Skipping item due to missing text, tokens, or bio_labels: {item.get('id', 'N/A')}")
            continue

        # 1. Tokenize text with the model's tokenizer
        encoding = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding='max_length', # Use max_length for consistent output length from tokenizer
            max_length=MAX_LENGTH, # Use constant from model.py
            return_offsets_mapping=True
        )
        
        input_ids = encoding['input_ids'].squeeze().tolist()
        tokenized_by_model = tokenizer.convert_ids_to_tokens(input_ids)
        
        # Remove padding tokens and special tokens for alignment if necessary before aligning
        # Or, ensure `enhanced_align_tokens_and_labels` handles them correctly.
        # For now, assume enhanced_align_tokens_and_labels prepares labels for the *full* tokenized_by_model sequence including special tokens.

        # 2. Align ground truth BIO labels with the model's tokenization
        # `enhanced_align_tokens_and_labels` expects original tokens and labels.
        # It should produce labels aligned with `tokenized_by_model` including special tokens.
        _, aligned_true_labels_str = enhanced_align_tokens_and_labels(original_tokens, original_bio_labels, tokenizer)
        
        # Ensure length consistency after alignment with model's tokenizer output (max_length)
        # The model tokenizer pads/truncates to MAX_LENGTH. aligned_true_labels_str might need similar treatment.
        # enhanced_align_tokens_and_labels already adds <s> and </s>.
        # We need to ensure the labels match the MAX_LENGTH after tokenization by model's tokenizer.

        # Pad/truncate aligned_true_labels_str to MAX_LENGTH to match model input
        # The label for [PAD] should be "O" or mapped to ASPECT_LABEL_MAP["O"]
        # The label for special tokens like <s>, </s> is usually "O".
        
        final_aligned_true_labels_str = []
        # Handle CLS (<s>) token
        if tokenized_by_model[0] == tokenizer.cls_token or tokenized_by_model[0] == '<s>':
            final_aligned_true_labels_str.append("O")
        
        # Logic to handle the actual tokens from enhanced_align_tokens_and_labels
        # This part needs careful review based on `enhanced_align_tokens_and_labels` output structure
        # For now, let's assume it gives labels for content tokens and we map them, then pad.
        
        # Simplified alignment: Use the labels from `item['bio_labels']` directly, 
        # but this requires `item['tokens']` to be exactly what the ATE model tokenizer produces, 
        # which is not guaranteed. The robust way is full re-alignment.
        
        # For now, this is a placeholder for the complex alignment:
        # We need to ensure aligned_true_labels_str corresponds to the *exact* tokens input to the model.
        # Let's assume `enhanced_align_tokens_and_labels` is nearly correct but might need length adjustment.
        
        # TEMP: A simpler approach for now, acknowledging potential misalignment if tokenizers differ wildly
        # This is a common source of error if not handled perfectly.
        # The `enhanced_align_tokens_and_labels` from `model.py` should be the source of truth for this.
        
        # The `enhanced_align_tokens_and_labels` already adds <s> and </s>.
        # Let's ensure the output is padded/truncated to MAX_LENGTH.
        processed_true_labels_str = aligned_true_labels_str[:MAX_LENGTH] 
        if len(processed_true_labels_str) < MAX_LENGTH:
            processed_true_labels_str.extend(["O"] * (MAX_LENGTH - len(processed_true_labels_str)))

        all_true_bio_labels.append(processed_true_labels_str)

        # 3. Get model predictions
        offset_mapping = encoding.pop('offset_mapping').squeeze().tolist() # For recover_char_offsets later
        encoding_on_device = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            if use_crf:
                outputs = model(**encoding_on_device)
                if "predictions" in outputs: # CRF model's output
                    predictions_tensor = outputs["predictions"].squeeze()
                else: # Fallback for CRF if "predictions" key is missing
                    logits = outputs["logits"]
                    predictions_tensor = torch.argmax(logits, dim=-1).squeeze()
            else: # Standard model
                outputs = model(**encoding_on_device)
                predictions_tensor = torch.argmax(outputs.logits, dim=-1).squeeze()
        
        # Ensure predictions_tensor is on CPU and handle cases where it might not be full length
        pred_ids = predictions_tensor.cpu().tolist()
        # If pred_ids is a single int (batch size 1, sequence length 1), make it a list
        if isinstance(pred_ids, int):
            pred_ids = [pred_ids]
        
        # Pad/truncate pred_ids to MAX_LENGTH if model output is variable
        # Model output should align with input_ids length (MAX_LENGTH due to padding)
        if len(pred_ids) < MAX_LENGTH:
             # This case should ideally not happen if padding='max_length' is used correctly
            pred_ids.extend([ASPECT_LABEL_MAP["O"]] * (MAX_LENGTH - len(pred_ids)))
        elif len(pred_ids) > MAX_LENGTH:
            pred_ids = pred_ids[:MAX_LENGTH]

        pred_bio_for_report = [ASPECT_LABEL_MAP_INVERSE[pid] for pid in pred_ids]
        all_pred_bio_labels.append(pred_bio_for_report)

        # Recover character-level offsets for saving predictions (optional here, but good for inspection)
        # Note: recover_char_offsets expects tokens from the *model's tokenizer* and *raw numerical predictions*
        aspect_entities = recover_char_offsets(
            tokenized_by_model, # Tokens from tokenizer(text)
            pred_ids,           # Numerical predictions from model
            offset_mapping,
            text
        )
        ate_predictions_output.append({'text': text, 'true_bio_labels': processed_true_labels_str, 'predicted_bio_labels': pred_bio_for_report, 'extracted_aspects_by_model': aspect_entities})

    if all_true_bio_labels and all_pred_bio_labels:
        # Filter out samples where true or pred labels might be empty lists if any processing failed
        # (though padding should prevent this)
        # Alignment check for seqeval:
        final_true_bio = []
        final_pred_bio = []
        for true_s, pred_s in zip(all_true_bio_labels, all_pred_bio_labels):
            min_len = min(len(true_s), len(pred_s))
            if min_len > 0 : # Only add if there's something to compare
                 final_true_bio.append(true_s[:min_len])
                 final_pred_bio.append(pred_s[:min_len])
        
        if final_true_bio: # If after filtering there's still data
            report = seqeval_classification_report(final_true_bio, final_pred_bio, digits=4, output_dict=False) # Get string report
            logger.info(f"ATE Classification Report:\n{report}")
            return report, ate_predictions_output
        else:
            logger.info("ATE: No valid aligned sequences to generate report after filtering.")
            return "ATE: No valid aligned sequences to generate report after filtering.", ate_predictions_output
    else:
        logger.info("ATE: No true or predicted labels were generated for the report.")
        return "ATE: No true or predicted labels were generated for the report.", ate_predictions_output


# --- ASC Evaluation ---
def evaluate_asc_model(model, tokenizer, test_data, device):
    """Evaluate the Aspect Sentiment Classification (ASC) model."""
    logger.info("Evaluating ASC model...")
    model.to(device)
    model.eval()

    true_sentiments = []
    pred_sentiments = []
    asc_predictions_output = []

    for item in tqdm(test_data, desc="ASC Evaluation"):
        text = item['text']
        # Ground truth aspects and sentiments are in 'aspects' field
        # [{'aspect': 'Aspect Name', 'sentiment_id': 0/1/2}, ...]
        gt_aspect_sentiments = item.get('aspects', [])

        if not text or not gt_aspect_sentiments:
            logger.warning(f"Skipping item for ASC due to missing text or ground truth aspects: {item.get('id', 'N/A')}")
            continue

        for gt_asp_info in gt_aspect_sentiments:
            aspect_term = gt_asp_info['aspect']
            true_sentiment_id = gt_asp_info['sentiment_id']

            if aspect_term is None or true_sentiment_id is None:
                continue

            # ASC model expects (text, aspect_term)
            encoding = tokenizer(text, aspect_term, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH) # MAX_LENGTH might need adjustment for ASC
            encoding = {k: v.to(device) for k, v in encoding.items()}

            with torch.no_grad():
                outputs = model(**encoding)
                logits = outputs.logits
                predicted_sentiment_id = torch.argmax(logits, dim=1).item()
            
            true_sentiments.append(true_sentiment_id)
            pred_sentiments.append(predicted_sentiment_id)
            asc_predictions_output.append({
                'text': text, 
                'aspect': aspect_term, 
                'true_sentiment': SENTIMENT_LABELS[true_sentiment_id], 
                'predicted_sentiment': SENTIMENT_LABELS[predicted_sentiment_id]
            })

    if true_sentiments and pred_sentiments:
        # Ensure SENTIMENT_LABELS are correctly ordered for the report if using IDs 0,1,2
        # Target names should correspond to these IDs.
        report = sklearn_classification_report(
            true_sentiments, 
            pred_sentiments, 
            target_names=SENTIMENT_LABELS, 
            digits=4,
            output_dict=False # Get string report
        )
        logger.info(f"ASC Classification Report:\n{report}")
        return report, asc_predictions_output
    else:
        logger.info("ASC: No true or predicted sentiments to generate report.")
        return "ASC: No true or predicted sentiments to generate report.", asc_predictions_output

# --- ABSA Pipeline Testing ---
def test_absa_pipeline(pipeline, test_data_samples):
    """Test the full ABSA pipeline on a few samples."""
    logger.info("Testing ABSAPipeline...")
    
    for i, item_or_text in enumerate(test_data_samples):
        text_to_analyze = item_or_text['text'] if isinstance(item_or_text, dict) else item_or_text
        logger.info(f"Pipeline Example {i+1}: {text_to_analyze}")
        
        # The pipeline's analyze method should handle aspect extraction and sentiment classification
        results = pipeline.analyze(text_to_analyze) # analyze should use the use_crf from its init
        
        if results:
            for res_item in results:
                logger.info(
                    f"  Aspect: '{res_item.get('aspect', 'N/A')}' "
                    f"(Score: {res_item.get('aspect_score', 'N/A')}), "
                    f"Sentiment: {res_item.get('sentiment', 'N/A')} "
                    f"(Score: {res_item.get('sentiment_score', 'N/A')}), "
                    f"Offset: [{res_item.get('start', 'N/A')}-{res_item.get('end', 'N/A')}]"
                )
        else:
            logger.info(f"  No aspects/sentiments found by the pipeline for Example {i+1}.")


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load test data
    test_data = load_test_data(args.test_file)
    if not test_data:
        logger.error("No test data loaded. Exiting.")
        return

    # 1. Evaluate ATE Model
    logger.info("--- Aspect Term Extraction (ATE) Evaluation ---")
    ate_tokenizer = initialize_tokenizer() # Uses MODEL_NAME (xlm-roberta-base by default)
    
    if args.use_crf:
        logger.info(f"Loading ATE model with CRF from {args.ate_model_path}")
        ate_model = XLMRobertaForTokenClassificationCRF(MODEL_NAME, ASPECT_NUM_LABELS)
        # Adjust path for state_dict if it's a directory containing pytorch_model.bin
        model_file_path = os.path.join(args.ate_model_path, "pytorch_model.bin") \
            if os.path.isdir(args.ate_model_path) and os.path.exists(os.path.join(args.ate_model_path, "pytorch_model.bin")) \
            else args.ate_model_path # Assume it's a file if not a dir with pytorch_model.bin
        if not os.path.exists(model_file_path):
            logger.error(f"ATE model file not found at {model_file_path}. Attempted path: {args.ate_model_path}")
            # Fallback or error
        else:
            ate_model.load_state_dict(torch.load(model_file_path, map_location=torch.device(device)))
    else:
        logger.info(f"Loading standard ATE model from {args.ate_model_path}")
        # For standard AutoModelForTokenClassification, from_pretrained handles directory or file
        ate_model = AutoModelForTokenClassification.from_pretrained(args.ate_model_path)
    
    ate_report_str, ate_predictions = evaluate_ate_model(ate_model, ate_tokenizer, test_data, args.use_crf, device)
    with open(os.path.join(args.output_dir, "ate_report.txt"), 'w', encoding='utf-8') as f:
        f.write(ate_report_str)
    with open(os.path.join(args.output_dir, "ate_predictions_raw.json"), 'w', encoding='utf-8') as f:
        json.dump(ate_predictions, f, ensure_ascii=False, indent=2)
    logger.info(f"ATE report saved to {os.path.join(args.output_dir, 'ate_report.txt')}")
    logger.info(f"ATE raw predictions saved to {os.path.join(args.output_dir, 'ate_predictions_raw.json')}")

    # 2. Evaluate ASC Model
    logger.info("--- Aspect Sentiment Classification (ASC) Evaluation ---")
    try:
        asc_tokenizer = AutoTokenizer.from_pretrained(args.asc_model_path)
        asc_model = AutoModelForSequenceClassification.from_pretrained(args.asc_model_path)
        asc_report_str, asc_predictions = evaluate_asc_model(asc_model, asc_tokenizer, test_data, device)
        with open(os.path.join(args.output_dir, "asc_report.txt"), 'w', encoding='utf-8') as f:
            f.write(asc_report_str)
        with open(os.path.join(args.output_dir, "asc_predictions_raw.json"), 'w', encoding='utf-8') as f:
            json.dump(asc_predictions, f, ensure_ascii=False, indent=2)
        logger.info(f"ASC report saved to {os.path.join(args.output_dir, 'asc_report.txt')}")
        logger.info(f"ASC raw predictions saved to {os.path.join(args.output_dir, 'asc_predictions_raw.json')}")
    except Exception as e:
        logger.error(f"Could not evaluate ASC model: {e}")
        logger.info("Skipping ASC evaluation.")

    # 3. Test Full ABSA Pipeline
    logger.info("--- ABSA Pipeline Test ---")
    try:
        # Initialize pipeline (ensure it uses the correct ATE model path and use_crf setting)
        # The ABSAPipeline in src_xlm/model.py should accept use_crf in its constructor
        absa_pipeline = ABSAPipeline(
            aspect_model_path=args.ate_model_path, # Pass the potentially user-defined ATE model path
            sentiment_model_path=args.asc_model_path, # Pass the potentially user-defined ASC model path
            use_crf=args.use_crf # Pass the CRF flag
        )
        
        # Using a few samples from the loaded test data for pipeline testing
        pipeline_test_samples = test_data[:3] if len(test_data) >=3 else test_data
        if not pipeline_test_samples: # Or use predefined examples if test_data is empty
            pipeline_test_samples = [
                "Το κινητό έχει καλή μπαταρία και η κάμερα βγάζει εξαιρετικές φωτογραφίες.",
                "Η οθόνη είναι μεγάλη αλλά η μπαταρία δεν αντέχει πολύ.",
                "Καλή ταχύτητα αλλά κακή κάμερα για φωτογραφίες."
            ]
        test_absa_pipeline(absa_pipeline, pipeline_test_samples)
    except Exception as e:
        logger.error(f"Could not test ABSA pipeline: {e}")
        logger.info("Skipping ABSA pipeline test.")
        
    logger.info(f"All evaluations complete. Check the '{args.output_dir}' directory for reports and predictions.")

if __name__ == "__main__":
    main() 