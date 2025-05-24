import json
import logging
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Import from model.py
from model import (
    # Constants
    ASPECT_MODEL_PATH, SENTIMENT_MODEL_PATH, SENTIMENT_LABELS, ASPECT_LABEL_MAP, ASPECT_LABEL_MAP_INVERSE,
    
    # Classes and functions
    ABSAPipeline, preprocess_text
)

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
                # Preprocess keyword for better matching
                keyword_proc = preprocess_text(keyword)
                if keyword_proc:
                    keyword_to_family[keyword_proc] = family
        
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
    # Preprocess keyword for better matching with our dictionary
    keyword_proc = preprocess_text(keyword).lower()
    
    # Direct match
    if keyword_proc in KEYWORD_TO_FAMILY:
        return KEYWORD_TO_FAMILY[keyword_proc]
    
    # Check if the keyword starts with or contains any key in the mapping
    for known_keyword, family in KEYWORD_TO_FAMILY.items():
        if keyword_proc.startswith(known_keyword) or known_keyword in keyword_proc:
            return family
    
    return "Unknown"

def main():
    parser = argparse.ArgumentParser(description='Inference script for XLM-RoBERTa ABSA')
    parser.add_argument('--text', type=str, required=True, help='Input text for analysis')
    parser.add_argument('--aspect_model', type=str, default=ASPECT_MODEL_PATH, 
                        help='Path to the aspect extraction model')
    parser.add_argument('--sentiment_model', type=str, default=SENTIMENT_MODEL_PATH,
                        help='Path to the sentiment classification model')
    parser.add_argument('--confidence', type=float, default=0.05, help='Confidence threshold for aspect extraction (default: 0.05)')
    
    args = parser.parse_args()
    
    # Use the standard ABSAPipeline
    try:
        # Create pipeline
        pipeline = ABSAPipeline(args.aspect_model, args.sentiment_model)
        results = pipeline.analyze(args.text)
        
        if results:
            logger.info(f"Analysis results for text: {args.text}")
            for i, result in enumerate(results):
                aspect_family = get_aspect_family(result['aspect'])
                sentiment_emoji = "😊" if result['sentiment'] == "positive" else "😐" if result['sentiment'] == "neutral" else "😞"
                logger.info(f"Aspect {i+1}: {aspect_family} (keyword: {result['aspect']}) - {result['sentiment']} {sentiment_emoji} ({result['sentiment_score']:.2f})")
        else:
            logger.info(f"No aspects found in: {args.text}")
            
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 