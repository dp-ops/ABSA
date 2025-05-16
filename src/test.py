import json
import logging
import argparse
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import re

from model import (
    # Constants
    ASPECT_MODEL_PATH, SENTIMENT_MODEL_PATH, SENTIMENT_LABELS,
    
    # Classes and functions
    initialize_tokenizer,
    load_aspect_dataset,
    load_sentiment_dataset,
    compute_ate_metrics,
    compute_asc_metrics,
    ABSAPipeline,
    load_and_process_example,
    evaluate_predictions
)

from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, Trainer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load aspect keywords from JSON file
def load_aspect_keywords(file_path="data/aspect_keywords_map.json"):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            aspect_keywords_map = json.load(f)
        
        # Create a reverse mapping from keyword to aspect family
        keyword_to_family = {}
        for family, keywords in aspect_keywords_map.items():
            for keyword in keywords:
                keyword_to_family[keyword.lower()] = family
        
        return aspect_keywords_map, keyword_to_family
    except Exception as e:
        logger.error(f"Error loading aspect keywords map: {str(e)}")
        return {}, {}

# Load the aspect keywords
ASPECT_KEYWORDS_MAP, KEYWORD_TO_FAMILY = load_aspect_keywords()

# Common adjectives that should not be treated as aspects
ADJECTIVES = {
    'καλη', 'καλή', 'καλο', 'καλό', 'κακη', 'κακή', 'κακο', 'κακό', 'ωραια', 'ωραία', 'ωραιο', 'ωραίο',
    'εξαιρετικη', 'εξαιρετική', 'εξαιρετικο', 'εξαιρετικό', 'τελεια', 'τέλεια', 'τελειο', 'τέλειο',
    'χαλια', 'χάλια', 'χαλιο', 'χάλιο', 'αργη', 'αργή', 'αργο', 'αργό', 'γρηγορη', 'γρήγορη', 'γρηγορο', 'γρήγορο',
    'δυνατη', 'δυνατή', 'δυνατο', 'δυνατό', 'αδυναμη', 'αδύναμη', 'αδυναμο', 'αδύναμο', 
    'μεγαλη', 'μεγάλη', 'μεγαλο', 'μεγάλο', 'μικρη', 'μικρή', 'μικρο', 'μικρό',
    'φθηνη', 'φθηνή', 'φθηνο', 'φθηνό', 'ακριβη', 'ακριβή', 'ακριβο', 'ακριβό'
}

# Non-aspect words that should be filtered out
NON_ASPECT_WORDS = {
    'ειναι', 'είναι', 'εχει', 'έχει', 'και', 'with', 'the', 'has', 'is', 'are', 'του', 'της', 'το',
    'για', 'για', 'απο', 'από', 'στον', 'στην', 'στο', 'στους', 'στις', 'στα', 'με', 'τα', 'τον', 'την'
}

def get_aspect_family(keyword):
    """Get the aspect family for a keyword, or 'Unknown' if not found"""
    keyword_lower = keyword.lower()
    
    # Direct match
    if keyword_lower in KEYWORD_TO_FAMILY:
        return KEYWORD_TO_FAMILY[keyword_lower]
    
    # Check if the keyword starts with or contains any key in the mapping
    for known_keyword, family in KEYWORD_TO_FAMILY.items():
        if keyword_lower.startswith(known_keyword) or known_keyword in keyword_lower:
            return family
    
    return "Unknown"

def parse_args():
    parser = argparse.ArgumentParser(description='Test ABSA models')
    parser.add_argument('--test_data', type=str, 
                        default="data/filtered_data/processed_aspect_data_test.json",
                        help='Path to test data file')
    parser.add_argument('--aspect_model', type=str, 
                        default=ASPECT_MODEL_PATH,
                        help='Path to aspect extraction model')
    parser.add_argument('--sentiment_model', type=str, 
                        default=SENTIMENT_MODEL_PATH,
                        help='Path to sentiment classification model')
    parser.add_argument('--output_dir', type=str, 
                        default="results",
                        help='Directory to save test results')
    parser.add_argument('--num_examples', type=int, 
                        default=5,
                        help='Number of examples to print in the detailed report')
    return parser.parse_args()

def evaluate_ate_model(model_path, test_data_path, output_dir):
    """Evaluate Aspect Term Extraction model"""
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return None
    
    logger.info(f"Evaluating ATE model from {model_path}")
    
    # Load model and tokenizer
    tokenizer = initialize_tokenizer()
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    # Load test dataset
    test_dataset = load_aspect_dataset(test_data_path, tokenizer)
    
    # Create trainer (without training)
    trainer = Trainer(
        model=model,
        compute_metrics=compute_ate_metrics
    )
    
    # Evaluate
    logger.info("Evaluating ATE model...")
    metrics = trainer.evaluate(test_dataset)
    
    # Save metrics
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/ate_test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"ATE Test metrics: {metrics}")
    return metrics

def evaluate_asc_model(model_path, test_data_path, output_dir):
    """Evaluate Aspect Sentiment Classification model"""
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return None
    
    logger.info(f"Evaluating ASC model from {model_path}")
    
    # Load model and tokenizer
    tokenizer = initialize_tokenizer()
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Load test dataset
    test_dataset = load_sentiment_dataset(test_data_path, tokenizer)
    
    # Create trainer (without training)
    trainer = Trainer(
        model=model,
        compute_metrics=compute_asc_metrics
    )
    
    # Evaluate
    logger.info("Evaluating ASC model...")
    metrics = trainer.evaluate(test_dataset)
    
    # Save metrics
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/asc_test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"ASC Test metrics: {metrics}")
    return metrics

def test_pipeline(pipeline_args, test_data_path, output_dir, num_examples=5):
    """Test the full ABSA pipeline on all examples in the test data"""
    try:
        # Initialize pipeline
        pipeline = ABSAPipeline(**pipeline_args)
        
        # Load all test data
        test_data = []
        with open(test_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    test_data.append(json.loads(line))
        
        logger.info(f"Testing pipeline on {len(test_data)} examples")
        
        # Process each example
        results = []
        for item in tqdm(test_data, desc="Processing test examples"):
            text = item.get('text', '')
            if not text:
                continue
                
            gold_aspects = [asp_info.get('aspect', '') for asp_info in item.get('aspects', [])]
            gold_aspects = [aspect for aspect in gold_aspects if aspect]  # Filter out empty aspects
            
            # Run the pipeline
            predictions = pipeline.analyze(text)
            
            # Assign family to each predicted aspect
            for pred in predictions:
                aspect_term = pred['aspect']
                # Add the aspect family if not already present
                if 'family' not in pred:
                    pred['family'] = get_aspect_family(aspect_term)
            
            # Store results
            result = {
                'text': text,
                'gold_aspects': gold_aspects,
                'predicted_aspects': predictions
            }
            results.append(result)
        
        # Evaluate overall performance
        metrics = evaluate_predictions(results)
        
        # Calculate additional metrics for aspects
        total_correct_aspects = 0
        total_predicted_aspects = 0
        total_gold_aspects = 0
        
        # For fuzzy matching aspects (more lenient evaluation)
        def normalize_text(text):
            """Normalize text for fuzzy comparison"""
            return re.sub(r'[^\w\s]', '', text.lower())
        
        for result in results:
            gold_aspects = set([normalize_text(aspect) for aspect in result['gold_aspects']])
            pred_aspects = set([normalize_text(pred['aspect']) for pred in result['predicted_aspects']])
            
            # Count exact and partial matches
            matches = 0
            for pred in pred_aspects:
                for gold in gold_aspects:
                    # Check for exact match or substring match
                    if pred == gold or pred in gold or gold in pred:
                        matches += 1
                        break
            
            total_correct_aspects += matches
            total_predicted_aspects += len(pred_aspects)
            total_gold_aspects += len(gold_aspects)
        
        # Calculate lenient metrics
        lenient_precision = total_correct_aspects / total_predicted_aspects if total_predicted_aspects > 0 else 0
        lenient_recall = total_correct_aspects / total_gold_aspects if total_gold_aspects > 0 else 0
        lenient_f1 = 2 * lenient_precision * lenient_recall / (lenient_precision + lenient_recall) if (lenient_precision + lenient_recall) > 0 else 0
        
        metrics.update({
            'lenient_precision': lenient_precision,
            'lenient_recall': lenient_recall,
            'lenient_f1': lenient_f1,
            'total_examples': len(results),
            'total_predicted_aspects': total_predicted_aspects,
            'total_gold_aspects': total_gold_aspects,
            'average_aspects_per_example': total_gold_aspects / len(results) if results else 0,
            'average_predicted_aspects_per_example': total_predicted_aspects / len(results) if results else 0
        })
        
        # Save metrics
        with open(f"{output_dir}/pipeline_test_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        
        # Save detailed results
        with open(f"{output_dir}/pipeline_test_results.json", "w") as f:
            json.dump(results, f, indent=4)
        
        # Print a sample of the examples
        print("\nExample Test Results:")
        for i, result in enumerate(results[:num_examples]):
            print(f"\nExample {i+1}:")
            print(f"Text: {result['text']}")
            
            print("\nGold Standard Aspects:")
            for aspect in result['gold_aspects']:
                print(f"  • {aspect}")
            
            print("\nPredicted Aspects and Sentiments:")
            if result['predicted_aspects']:
                for pred in result['predicted_aspects']:
                    sentiment_emoji = "😊" if pred['sentiment'] == "positive" else "😐" if pred['sentiment'] == "neutral" else "😞"
                    family_display = pred.get('family', 'Unknown')
                    print(f"  • {family_display} (keyword: {pred['aspect']}) → {pred['sentiment']} {sentiment_emoji} (confidence: {pred['sentiment_score']:.2f})")
            else:
                print("  No aspects predicted.")
        
        # Print overall metrics
        print("\nOverall Pipeline Evaluation Metrics:")
        print(f"  Exact Match Metrics:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1 Score: {metrics['f1']:.4f}")
        print(f"\n  Lenient Match Metrics (including partial matches):")
        print(f"    Precision: {metrics['lenient_precision']:.4f}")
        print(f"    Recall: {metrics['lenient_recall']:.4f}")
        print(f"    F1 Score: {metrics['lenient_f1']:.4f}")
        print(f"\n  Statistics:")
        print(f"    Total examples: {metrics['num_examples']}")
        print(f"    Total predicted aspects: {metrics['total_predicted_aspects']}")
        print(f"    Total gold standard aspects: {metrics['total_gold_aspects']}")
        print(f"    Average aspects per example: {metrics['average_aspects_per_example']:.2f}")
        print(f"    Average predicted aspects per example: {metrics['average_predicted_aspects_per_example']:.2f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error testing pipeline: {str(e)}")
        if 'pipeline' in locals():
            del pipeline
        import traceback
        traceback.print_exc()
        return None

def main():
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Test individual models
    ate_metrics = evaluate_ate_model(args.aspect_model, args.test_data, args.output_dir)
    asc_metrics = evaluate_asc_model(args.sentiment_model, args.test_data, args.output_dir)
    
    # Test pipeline
    pipeline_args = {
        "aspect_model_path": args.aspect_model,
        "sentiment_model_path": args.sentiment_model
    }
    
    pipeline_metrics = test_pipeline(
        pipeline_args, 
        args.test_data, 
        args.output_dir,
        args.num_examples
    )
    
    # Print summary
    print("\n" + "=" * 50)
    print("ABSA MODEL EVALUATION SUMMARY")
    print("=" * 50)
    
    if ate_metrics:
        print(f"\nAspect Extraction (ATE) Metrics:")
        print(f"  Precision: {ate_metrics.get('eval_precision', 'N/A')}")
        print(f"  Recall: {ate_metrics.get('eval_recall', 'N/A')}")
        print(f"  F1 Score: {ate_metrics.get('eval_f1', 'N/A')}")
    
    if asc_metrics:
        print(f"\nAspect Sentiment Classification (ASC) Metrics:")
        print(f"  Accuracy: {asc_metrics.get('eval_accuracy', 'N/A')}")
        print(f"  Macro F1: {asc_metrics.get('eval_macro_f1', 'N/A')}")
        print(f"  Negative F1: {asc_metrics.get('eval_neg_f1', 'N/A')}")
        print(f"  Neutral F1: {asc_metrics.get('eval_neu_f1', 'N/A')}")
        print(f"  Positive F1: {asc_metrics.get('eval_pos_f1', 'N/A')}")
    
    if pipeline_metrics:
        print(f"\nEnd-to-End Pipeline Metrics:")
        print(f"  Exact Match:")
        print(f"    Precision: {pipeline_metrics['precision']:.4f}")
        print(f"    Recall: {pipeline_metrics['recall']:.4f}")
        print(f"    F1 Score: {pipeline_metrics['f1']:.4f}")
        print(f"  Lenient Match (including partial matches):")
        print(f"    Precision: {pipeline_metrics['lenient_precision']:.4f}")
        print(f"    Recall: {pipeline_metrics['lenient_recall']:.4f}")
        print(f"    F1 Score: {pipeline_metrics['lenient_f1']:.4f}")
    
    print("\nDetailed results saved to:", args.output_dir)
    print("=" * 50)

if __name__ == "__main__":
    main() 