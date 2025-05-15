import json
import logging
import argparse
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
# from align_tokens import align_tokens_and_labels, convert_aligned_labels_to_ids
from model import ASPECT_LABEL_MAP, ASPECT_LABEL_MAP_INVERSE

# Import from model.py
from model import (
    # Constants
    ASPECT_MODEL_PATH, SENTIMENT_MODEL_PATH, SENTIMENT_LABELS,
    
    # Classes and functions
    ABSAPipeline,
    load_and_process_example,
    evaluate_predictions
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

def debug_ate(text, model_path, confidence_threshold=0.05):
    """Debug ATE by showing token-level predictions with lower confidence threshold"""
    logger.info(f"--- Debugging ATE for text: {text} ---")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    # Tokenize the input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # Get the predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits
    
    # Convert to probabilities
    probs = torch.nn.functional.softmax(predictions, dim=2)
    # Get the predicted labels
    pred_labels = torch.argmax(probs, dim=2)
    
    # Convert to numpy for easier handling
    probs_np = probs.detach().numpy()[0]
    pred_labels_np = pred_labels.detach().numpy()[0]
    
    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Print token-level predictions with probabilities
    logger.info("\nToken-level ATE predictions with probabilities:")
    for i, (token, label_id, token_probs) in enumerate(zip(tokens, pred_labels_np, probs_np)):
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue
        
        label = ASPECT_LABEL_MAP_INVERSE.get(label_id, f"UNKNOWN-{label_id}")
        prob = token_probs[label_id]
        
        # Check if token is potentially an aspect based on keyword matching
        potential_aspect = False
        aspect_family = None
        cleaned_token = token.replace("##", "")
        
        # Check if it's in our keyword mapping
        for family, keywords in ASPECT_KEYWORDS_MAP.items():
            if cleaned_token.lower() in [k.lower() for k in keywords] or token.lower() in [k.lower() for k in keywords]:
                potential_aspect = True
                aspect_family = family
                break
        
        # Set label color and format based on probability and keyword matching
        if (label != "O" and prob >= confidence_threshold) or potential_aspect:
            if potential_aspect and label == "O":
                label_str = f"* POTENTIAL-ASP ({prob:.3f} for O) * [Family: {aspect_family or 'Unknown'}]"
            else:
                label_str = f"* {label} ({prob:.3f}) *"
        else:
            label_str = f"{label} ({prob:.3f})"
            
        logger.info(f"  {token} -> {label_str}")
    
    # Extract aspects with confidence above threshold, also considering keyword matches
    aspects = []
    i = 0
    while i < len(tokens):
        if tokens[i] in ['[CLS]', '[SEP]', '[PAD]']:
            i += 1
            continue
            
        # Check if it's a predicted aspect or a keyword match
        token_is_aspect = False
        token_family = None
        
        # Check if it's predicted as an aspect
        if pred_labels_np[i] == ASPECT_LABEL_MAP['B-ASP'] and probs_np[i][pred_labels_np[i]] >= confidence_threshold:
            token_is_aspect = True
            confidence = float(probs_np[i][pred_labels_np[i]])
        
        # If not predicted, check if it's a keyword match for any aspect type
        if not token_is_aspect:
            token_text = tokens[i].replace("##", "").lower()
            for family, keywords in ASPECT_KEYWORDS_MAP.items():
                if token_text in [k.lower() for k in keywords] or tokens[i].lower() in [k.lower() for k in keywords]:
                    token_is_aspect = True
                    token_family = family
                    confidence = 0.5  # Default confidence for keyword matches
                    break
        
        if token_is_aspect:
            # Found the beginning of an aspect
            aspect_start = i
            aspect_end = i
            
            # Look for continuation (I-ASP) or keyword matches
            j = i + 1
            while j < len(tokens) and j < len(pred_labels_np):
                if tokens[j] in ['[CLS]', '[SEP]', '[PAD]']:
                    j += 1
                    continue
                
                token_continue = False
                
                # Check if marked as I-ASP
                if pred_labels_np[j] == ASPECT_LABEL_MAP['I-ASP'] and probs_np[j][pred_labels_np[j]] >= confidence_threshold:
                    token_continue = True
                
                # If part of the same word (subword token starting with ##)
                if tokens[j].startswith('##'):
                    token_continue = True
                
                if token_continue:
                    aspect_end = j
                    j += 1
                else:
                    break
            
            # Extract the aspect text
            aspect_tokens = tokens[aspect_start:aspect_end+1]
            # Remove ## from subword tokens
            aspect_text = ''.join([t.replace('##', '') for t in aspect_tokens])
            
            # Skip if it's in the non-aspect words or adjectives list
            if aspect_text.lower() in NON_ASPECT_WORDS or aspect_text.lower() in ADJECTIVES:
                i = aspect_end + 1
                continue
            
            # Determine the aspect family
            if token_family is None:
                token_family = get_aspect_family(aspect_text)
                
            aspects.append({
                'text': aspect_text,
                'family': token_family,
                'start_token': aspect_start,
                'end_token': aspect_end,
                'confidence': confidence
            })
            
            i = aspect_end + 1
        else:
            i += 1
    
    if aspects:
        logger.info("\nExtracted aspects:")
        for i, aspect in enumerate(aspects):
            family = aspect.get('family', 'Unknown')
            logger.info(f"  {i+1}. {family} (keyword: {aspect['text']}, confidence: {aspect['confidence']:.2f})")
    else:
        logger.info("\nNo aspects found above confidence threshold or through keyword matching.")
        
    logger.info("--- End ATE Debug ---")
    return aspects

def main():
    parser = argparse.ArgumentParser(description='Inference script for ABSA')
    parser.add_argument('--text', type=str, required=True, help='Input text for analysis')
    parser.add_argument('--aspect_model', type=str, default='saved_models/aspect_extractor_model', 
                        help='Path to the aspect extraction model')
    parser.add_argument('--sentiment_model', type=str, default='saved_models/aspect_sentiment_model',
                        help='Path to the sentiment classification model')
    parser.add_argument('--debug_ate', action='store_true', help='Debug the ATE predictions')
    parser.add_argument('--confidence', type=float, default=0.05, help='Confidence threshold for aspect extraction (default: 0.05). Note: Adjectives are filtered by default.')
    
    args = parser.parse_args()
    
    # For debugging
    if args.debug_ate:
        debug_ate(args.text, args.aspect_model, args.confidence)
        return
    
    # For regular analysis, use the ABSAPipeline
    try:
        # Create a custom ABSAPipeline with improved aspect extraction
        class EnhancedABSAPipeline(ABSAPipeline):
            # Non-aspect words that should be filtered out even if the model detected them
            NON_ASPECT_WORDS = NON_ASPECT_WORDS
            
            # Common adjectives that should not be treated as aspects
            ADJECTIVES = ADJECTIVES
            
            def extract_aspects(self, text, confidence_threshold=0.05):
                """Enhanced aspect extraction with keyword matching and lower confidence threshold"""
                if not text or not text.strip():
                    logger.warning("Empty text provided for aspect extraction")
                    return []
                
                try:
                    # First try the model-based extraction with a lower threshold
                    model_aspects = super().extract_aspects(text, confidence_threshold)
                    
                    # Filter out common words that aren't actually aspects
                    filtered_aspects = []
                    for aspect in model_aspects:
                        aspect_word = aspect['word'].lower()
                        # Filter out very short words (1-2 characters)
                        if len(aspect_word) <= 2:
                            continue
                        # Filter out common words that shouldn't be aspects
                        if aspect_word not in self.NON_ASPECT_WORDS and aspect_word not in self.ADJECTIVES:
                            # Add the aspect family
                            aspect_family = get_aspect_family(aspect_word)
                            aspect['family'] = aspect_family
                            filtered_aspects.append(aspect)
                    
                    # Then try keyword-based extraction for common aspects
                    # Tokenize the input text for keyword matching
                    words = text.lower().split()
                    
                    keyword_aspects = []
                    for word in words:
                        # Skip short words, common non-aspect words and adjectives
                        if len(word) <= 2 or word in self.NON_ASPECT_WORDS or word in self.ADJECTIVES:
                            continue
                            
                        for family, keywords in ASPECT_KEYWORDS_MAP.items():
                            if word.lower() in [k.lower() for k in keywords]:
                                # Find this word in the original text
                                start_char = text.lower().find(word)
                                if start_char >= 0:
                                    end_char = start_char + len(word)
                                    
                                    # Check if this aspect is already covered by model predictions
                                    already_covered = False
                                    for aspect in filtered_aspects:
                                        # Check for overlap
                                        if (aspect['start'] <= start_char < aspect['end'] or 
                                            aspect['start'] < end_char <= aspect['end'] or
                                            start_char <= aspect['start'] < aspect['end'] <= end_char):
                                            already_covered = True
                                            break
                                    
                                    if not already_covered:
                                        keyword_aspects.append({
                                            'word': word,
                                            'family': family,
                                            'score': 0.5,  # Default confidence for keyword matches
                                            'start': start_char,
                                            'end': end_char
                                        })
                    
                    # Combine both lists
                    return filtered_aspects + keyword_aspects
                
                except Exception as e:
                    logger.error(f"Error in enhanced aspect extraction: {str(e)}")
                    return []
                    
            def analyze(self, text):
                """Full ABSA pipeline with aspect families: extract aspects and determine their sentiment"""
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
                    
                    aspect_family = aspect.get('family', get_aspect_family(aspect_term))
                    
                    results.append({
                        'aspect': aspect_term,
                        'family': aspect_family,
                        'sentiment': SENTIMENT_LABELS[sentiment['label']],
                        'sentiment_score': sentiment['score'],
                        'aspect_score': aspect['score'],
                        'start': aspect.get('start', -1),
                        'end': aspect.get('end', -1)
                    })
                
                return results
        
        # Use the enhanced pipeline with lower confidence threshold
        pipeline = EnhancedABSAPipeline(args.aspect_model, args.sentiment_model)
        results = pipeline.analyze(args.text)
        
        if results:
            logger.info(f"Analysis results for text: {args.text}")
            for i, result in enumerate(results):
                family_display = result.get('family', 'Unknown')
                logger.info(f"Aspect {i+1}: {family_display} (keyword: {result['aspect']}) - {result['sentiment']} ({result['sentiment_score']:.2f})")
        else:
            logger.info(f"No aspects found in: {args.text}")
            
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")

if __name__ == "__main__":
    main() 