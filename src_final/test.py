import json
import logging
import argparse
import os
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report as sklearn_classification_report
from sklearn.metrics import multilabel_confusion_matrix, hamming_loss, jaccard_score

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from model import (
    # Constants
    ATC_MODEL_PATH, ASC_MODEL_PATH, MODEL_NAME,
    ASPECT_CATEGORIES, ASPECT_TO_ID, ID_TO_ASPECT,
    SENTIMENT_LABELS, SENTIMENT_TO_ID, ID_TO_SENTIMENT,
    MAX_LENGTH,
    
    # Classes and functions
    initialize_tokenizer, MultiLabelBertForAspectClassification,
    ABSAPipeline
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Test Greek BERT ATC and ASC models, and ABSA pipeline")
    parser.add_argument('--test_file', default='data/f_data_bert/test_data.json',
                        help='Path to test data file (must contain text_processed and aspects_present)')
    parser.add_argument('--atc_model_path', default=ATC_MODEL_PATH,
                        help='Path to trained ATC model')
    parser.add_argument('--asc_model_path', default=ASC_MODEL_PATH,
                        help='Path to trained ASC model')
    parser.add_argument('--output_dir', default='results',
                        help='Directory to save prediction files and reports')
    parser.add_argument('--aspect_threshold', type=float, default=0.5,
                        help='Threshold for aspect classification (default: 0.5)')
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

# --- ATC Evaluation ---
def evaluate_atc_model(model, tokenizer, test_data, device, threshold=0.5):
    """Evaluate the Aspect Term Classification (ATC) model."""
    logger.info("Evaluating ATC model...")
    model.to(device)
    model.eval()

    all_true_labels = []
    all_pred_labels = []
    atc_predictions_output = []

    for item in tqdm(test_data, desc="ATC Evaluation"):
        text = item['text_processed']
        aspects_present = item.get('aspects_present', [])

        if not text:
            logger.warning(f"Skipping item due to missing text: {item.get('id', 'N/A')}")
            continue

        # Create true multi-label vector
        true_labels = [0.0] * len(ASPECT_CATEGORIES)
        for aspect_info in aspects_present:
            aspect_name = aspect_info['aspect_category']
            if aspect_name in ASPECT_TO_ID:
                true_labels[ASPECT_TO_ID[aspect_name]] = 1.0

        all_true_labels.append(true_labels)

        # Get model predictions
        encoding = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH
        )
        
        encoding_on_device = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**encoding_on_device)
            logits = outputs.logits
            
            # Apply sigmoid and threshold
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_labels = (probs > threshold).astype(float)

        all_pred_labels.append(pred_labels.tolist())

        # Create readable prediction output
        predicted_aspects = []
        true_aspects = []
        
        for i, (prob, pred, true) in enumerate(zip(probs, pred_labels, true_labels)):
            aspect_name = ASPECT_CATEGORIES[i]
            if pred == 1:
                predicted_aspects.append({'aspect': aspect_name, 'score': float(prob)})
            if true == 1:
                true_aspects.append(aspect_name)

        atc_predictions_output.append({
            'text': text,
            'true_aspects': true_aspects,
            'predicted_aspects': predicted_aspects,
            'true_labels': true_labels,
            'predicted_labels': pred_labels.tolist(),
            'probabilities': probs.tolist()
        })

    # Calculate metrics
    if all_true_labels and all_pred_labels:
        true_labels_np = np.array(all_true_labels)
        pred_labels_np = np.array(all_pred_labels)
        
        # Calculate various multi-label metrics
        hamming = hamming_loss(true_labels_np, pred_labels_np)
        jaccard = jaccard_score(true_labels_np, pred_labels_np, average='samples', zero_division=0)
        
        # Per-aspect metrics
        report_dict = {}
        for i, aspect in enumerate(ASPECT_CATEGORIES):
            y_true = true_labels_np[:, i]
            y_pred = pred_labels_np[:, i]
            
            if y_true.sum() > 0:  # Only calculate if there are positive examples
                report = sklearn_classification_report(
                    y_true, y_pred, 
                    target_names=['Not Present', 'Present'],
                    output_dict=True,
                    zero_division=0
                )
                report_dict[aspect] = report

        # Overall metrics
        overall_report = sklearn_classification_report(
            true_labels_np, pred_labels_np,
            target_names=[f'Aspect_{i}' for i in range(len(ASPECT_CATEGORIES))],
            output_dict=False,
            zero_division=0
        )
        
        logger.info(f"ATC Hamming Loss: {hamming:.4f}")
        logger.info(f"ATC Jaccard Score: {jaccard:.4f}")
        logger.info(f"ATC Classification Report:\n{overall_report}")
        
        return {
            'hamming_loss': hamming,
            'jaccard_score': jaccard,
            'overall_report': overall_report,
            'per_aspect_reports': report_dict
        }, atc_predictions_output
    else:
        logger.info("ATC: No true or predicted labels were generated for the report.")
        return "ATC: No true or predicted labels were generated for the report.", atc_predictions_output

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
        text = item['text_processed']
        aspects_present = item.get('aspects_present', [])

        # if not text or not aspects_present:
        #     logger.warning(f"Skipping item for ASC due to missing text or aspects: {item.get('id', 'N/A')}")
        #     continue

        for aspect_info in aspects_present:
            aspect_name = aspect_info['aspect_category']
            true_sentiment_id = aspect_info['sentiment_id']

            if aspect_name is None or true_sentiment_id is None:
                continue

            # ASC model expects (text, aspect_name)
            encoding = tokenizer(
                text, aspect_name, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=MAX_LENGTH
            )
            encoding = {k: v.to(device) for k, v in encoding.items()}

            with torch.no_grad():
                outputs = model(**encoding)
                logits = outputs.logits
                predicted_sentiment_id = torch.argmax(logits, dim=1).item()
                
                # Get prediction probabilities
                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            
            true_sentiments.append(true_sentiment_id)
            pred_sentiments.append(predicted_sentiment_id)
            
            asc_predictions_output.append({
                'text': text,
                'aspect': aspect_name,
                'true_sentiment': SENTIMENT_LABELS[true_sentiment_id],
                'predicted_sentiment': SENTIMENT_LABELS[predicted_sentiment_id],
                'prediction_probabilities': {
                    label: float(prob) for label, prob in zip(SENTIMENT_LABELS, probs)
                }
            })

    if true_sentiments and pred_sentiments:
        report = sklearn_classification_report(
            true_sentiments, 
            pred_sentiments, 
            target_names=SENTIMENT_LABELS, 
            digits=4,
            output_dict=False
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
        text_to_analyze = item_or_text['text_processed'] if isinstance(item_or_text, dict) else item_or_text
        logger.info(f"Pipeline Example {i+1}: {text_to_analyze}")
        
        # The pipeline's analyze method should handle aspect extraction and sentiment classification
        results = pipeline.analyze(text_to_analyze)
        
        if results:
            logger.info(f"  Detected {len(results)} aspects:")
            for res_item in results:
                aspect = res_item.get('aspect', 'N/A')
                sentiment = res_item.get('sentiment', 'N/A')
                
                # Handle the score formatting safely - use the correct keys from ABSAPipeline
                aspect_confidence = res_item.get('aspect_confidence', 'N/A')
                sentiment_confidence = res_item.get('sentiment_confidence', 'N/A')
                
                # Format scores only if they are numeric
                if isinstance(aspect_confidence, (int, float)):
                    aspect_score_str = f"{aspect_confidence:.4f}"
                else:
                    aspect_score_str = str(aspect_confidence)
                    
                if isinstance(sentiment_confidence, (int, float)):
                    sentiment_score_str = f"{sentiment_confidence:.4f}"
                else:
                    sentiment_score_str = str(sentiment_confidence)
                
                logger.info(f"    - Aspect: '{aspect}' (Score: {aspect_score_str}) → Sentiment: {sentiment} (Score: {sentiment_score_str})")
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

    # 1. Evaluate ATC Model
    logger.info("--- Aspect Term Classification (ATC) Evaluation ---")
    try:
        atc_tokenizer = initialize_tokenizer()
        logger.info(f"Loading ATC model from {args.atc_model_path}")
        
        # Load the ATC model the same way as in ABSAPipeline
        atc_model = AutoModelForSequenceClassification.from_pretrained(args.atc_model_path)
        logger.info("Loaded ATC model successfully")
        
        atc_report, atc_predictions = evaluate_atc_model(
            atc_model, atc_tokenizer, test_data, device, args.aspect_threshold
        )
        
        # Save ATC results
        atc_report_file = os.path.join(args.output_dir, "atc_report.txt")
        with open(atc_report_file, 'w', encoding='utf-8') as f:
            if isinstance(atc_report, dict):
                f.write(f"Hamming Loss: {atc_report['hamming_loss']:.4f}\n")
                f.write(f"Jaccard Score: {atc_report['jaccard_score']:.4f}\n\n")
                f.write("Overall Classification Report:\n")
                f.write(atc_report['overall_report'])
                f.write("\n\nPer-Aspect Reports:\n")
                for aspect, report in atc_report['per_aspect_reports'].items():
                    f.write(f"\n{aspect}:\n")
                    f.write(str(report))
            else:
                f.write(str(atc_report))
        
        with open(os.path.join(args.output_dir, "atc_predictions_raw.json"), 'w', encoding='utf-8') as f:
            json.dump(atc_predictions, f, ensure_ascii=False, indent=2)
        
        logger.info(f"ATC report saved to {atc_report_file}")
        logger.info(f"ATC raw predictions saved to {os.path.join(args.output_dir, 'atc_predictions_raw.json')}")
        
    except Exception as e:
        logger.error(f"Could not evaluate ATC model: {e}")
        logger.info("Skipping ATC evaluation.")

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
        # Initialize pipeline
        absa_pipeline = ABSAPipeline(
            atc_model_path=args.atc_model_path,
            asc_model_path=args.asc_model_path
        )
        
        # Using a few samples from the loaded test data for pipeline testing
        pipeline_test_samples = test_data[:3] if len(test_data) >= 3 else test_data
        if not pipeline_test_samples:  # Or use predefined examples if test_data is empty
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
