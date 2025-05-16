import json
import logging
import argparse
# import os
import torch
# import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification#, pipeline
# import unicodedata
# import re

# Import from model_r.py
from model_r import (
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

def debug_ate(text, model_path, confidence_threshold=0.05):
    """Debug ATE by showing token-level predictions with lower confidence threshold"""
    logger.info(f"--- Debugging ATE for text: {text} ---")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    # Preprocess text for RoBERTa
    text_proc = text  # We'll keep the original for better output display
    
    # Tokenize the input
    inputs = tokenizer(text_proc, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
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
        if token in ['<s>', '</s>', '<pad>']:
            continue
        
        label = ASPECT_LABEL_MAP_INVERSE.get(label_id, f"UNKNOWN-{label_id}")
        prob = token_probs[label_id]
        
        # Check if token is potentially an aspect based on keyword matching
        potential_aspect = False
        aspect_family = None
        
        # Clean token for better matching - RoBERTa uses Ġ to indicate start of words
        cleaned_token = token.replace("Ġ", "")
        cleaned_token_proc = preprocess_text(cleaned_token).lower()
        
        # Check if it's in our keyword mapping
        for family, keywords in ASPECT_KEYWORDS_MAP.items():
            keywords_proc = [preprocess_text(k).lower() for k in keywords]
            if cleaned_token_proc in keywords_proc or any(cleaned_token_proc in k_proc for k_proc in keywords_proc if k_proc):
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
        if tokens[i] in ['<s>', '</s>', '<pad>']:
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
            # For RoBERTa, we need to handle the special tokens
            token_text = tokens[i].replace("Ġ", "").lower()
            token_text_proc = preprocess_text(token_text).lower()
            
            for family, keywords in ASPECT_KEYWORDS_MAP.items():
                keywords_proc = [preprocess_text(k).lower() for k in keywords]
                if token_text_proc in keywords_proc or any(token_text_proc in k_proc for k_proc in keywords_proc if k_proc):
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
                if tokens[j] in ['<s>', '</s>', '<pad>']:
                    j += 1
                    continue
                
                token_continue = False
                
                # Check if marked as I-ASP
                if pred_labels_np[j] == ASPECT_LABEL_MAP['I-ASP'] and probs_np[j][pred_labels_np[j]] >= confidence_threshold:
                    token_continue = True
                
                # For RoBERTa, if token doesn't start with Ġ, it's part of the previous word
                if not tokens[j].startswith('Ġ'):
                    token_continue = True
                
                if token_continue:
                    aspect_end = j
                    j += 1
                else:
                    break
            
            # Extract the aspect text
            aspect_tokens = tokens[aspect_start:aspect_end+1]
            # RoBERTa uses Ġ to indicate start of words, remove them and join
            aspect_text = ''.join([t.replace('Ġ', ' ') for t in aspect_tokens]).strip()
            
            # Skip if it's in the non-aspect words or adjectives list
            aspect_text_proc = preprocess_text(aspect_text).lower()
            if aspect_text_proc in [preprocess_text(w).lower() for w in NON_ASPECT_WORDS] or aspect_text_proc in [preprocess_text(w).lower() for w in ADJECTIVES]:
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
    parser = argparse.ArgumentParser(description='Inference script for RoBERTa ABSA')
    parser.add_argument('--text', type=str, required=True, help='Input text for analysis')
    parser.add_argument('--aspect_model', type=str, default=ASPECT_MODEL_PATH, 
                        help='Path to the aspect extraction model')
    parser.add_argument('--sentiment_model', type=str, default=SENTIMENT_MODEL_PATH,
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
                        aspect_word_proc = preprocess_text(aspect_word).lower()
                        
                        # Filter out very short words (1-2 characters)
                        if len(aspect_word) <= 2:
                            continue
                        # Filter out common words that shouldn't be aspects
                        if not any(aspect_word_proc == preprocess_text(w).lower() for w in self.NON_ASPECT_WORDS) and \
                           not any(aspect_word_proc == preprocess_text(w).lower() for w in self.ADJECTIVES):
                            # Add the aspect family
                            aspect_family = get_aspect_family(aspect_word)
                            aspect['family'] = aspect_family
                            filtered_aspects.append(aspect)
                    
                    # Then try keyword-based extraction for common aspects
                    # Tokenize and preprocess the input text for keyword matching
                    text_proc = preprocess_text(text).lower()
                    words = text_proc.split()
                    
                    keyword_aspects = []
                    for word in words:
                        # Skip short words, common non-aspect words and adjectives
                        if len(word) <= 2 or \
                           any(word == preprocess_text(w).lower() for w in self.NON_ASPECT_WORDS) or \
                           any(word == preprocess_text(w).lower() for w in self.ADJECTIVES):
                            continue
                            
                        for family, keywords in ASPECT_KEYWORDS_MAP.items():
                            keywords_proc = [preprocess_text(k).lower() for k in keywords]
                            if word in keywords_proc or any(word in k_proc for k_proc in keywords_proc if k_proc):
                                # Find this word in the original text (approximate matching)
                                start_char = -1
                                end_char = -1
                                
                                # Try to find the original word in text
                                original_words = text.lower().split()
                                for i, orig_word in enumerate(original_words):
                                    if preprocess_text(orig_word).lower() == word:
                                        # Calculate approximate character position
                                        start_char = text.lower().find(orig_word)
                                        if start_char >= 0:
                                            end_char = start_char + len(orig_word)
                                            break
                                
                                if start_char >= 0:
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
                                        # Find the original word in text for better display
                                        original_word = text[start_char:end_char] if start_char >= 0 else word
                                        
                                        keyword_aspects.append({
                                            'word': original_word,
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
                sentiment_emoji = "😊" if result['sentiment'] == "positive" else "😐" if result['sentiment'] == "neutral" else "😞"
                logger.info(f"Aspect {i+1}: {family_display} (keyword: {result['aspect']}) - {result['sentiment']} {sentiment_emoji} ({result['sentiment_score']:.2f})")
        else:
            logger.info(f"No aspects found in: {args.text}")
            
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 