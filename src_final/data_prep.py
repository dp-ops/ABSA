import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import re
import logging
from tqdm import tqdm
import argparse
import unicodedata
import random
import time

# Import translation library
try:
    from googletrans import Translator
    TRANSLATION_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Google Translate library available for data augmentation")
except ImportError:
    TRANSLATION_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Google Translate library not available. Install with: pip install googletrans==4.0.0-rc1")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the tokenizer for BERT-base-greek
MODEL_NAME = "nlpaueb/bert-base-greek-uncased-v1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Preprocessing function for BERT-base-greek
def strip_accents_and_lowercase(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn').lower()

def preprocess_text(text):
    """
    Preprocess text for BERT-base-greek:
    - Deaccent and lowercase
    - Basic cleaning (remove excessive whitespace)
    """
    if not text:
        return ""
    text = strip_accents_and_lowercase(str(text))
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Tokenization function with offset mapping
def tokenize_text(text):
    """
    Tokenize the preprocessed text and return tokens with offsets.
    """
    encoded = tokenizer(text, return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    offsets = encoded["offset_mapping"]
    return tokens, offsets

# Function to extract aspects and sentiments from CSV row
def extract_rated_aspects(row):
    """Extract aspects and sentiments from CSV row."""
    aspect_columns = [
        "Ποιότητα κλήσης", "Φωτογραφίες", "Καταγραφή Video", "Ταχύτητα",
        "Ανάλυση οθόνης", "Μπαταρία", "Σχέση ποιότητας τιμής", "Μουσική"
    ]
    extracted = []
    for col_name in aspect_columns:
        if col_name in row and pd.notna(row[col_name]) and str(row[col_name]).strip() in ['-1', '0', '1']:  # Exclude '-'
            sentiment_str = str(row[col_name]).strip()
            sentiment_val = -1 if sentiment_str == '-1' else int(sentiment_str)
            sentiment_str_val = "positive" if sentiment_val == 1 else "negative" if sentiment_val == -1 else "neutral"
            
            extracted.append({
                "aspect": col_name,
                "sentiment_val": sentiment_val,
                "sentiment_str": sentiment_str_val
            })
    return extracted

def find_aspect_terms_in_text(text, aspect_name, keywords):
    """
    Find aspect terms in text.
    Args:
        text: The text to search in (already preprocessed)
        aspect_name: The name of the aspect
        keywords: List of keywords related to the aspect
    Returns:
        List of (start_pos, end_pos, aspect_name) tuples
    """
    spans = []
    processed_keywords = []
    for keyword in keywords:
        processed_keyword = preprocess_text(keyword) # Preprocess keywords similarly
        if processed_keyword and len(processed_keyword) >= 3:
            processed_keywords.append(processed_keyword)

    # First try exact matches of the aspect name (preprocessed)
    aspect_name_processed = preprocess_text(aspect_name)
    start_idx = text.find(aspect_name_processed)
    if start_idx != -1:
        end_idx = start_idx + len(aspect_name_processed)
        spans.append((start_idx, end_idx, aspect_name))
        logger.debug(f"Direct aspect name match: '{aspect_name_processed}' at {start_idx}-{end_idx}")

    # Then try keyword matching
    for keyword_processed in processed_keywords:
        if not keyword_processed or len(keyword_processed) < 3:
            continue
        
        start_pos = 0
        while True:
            start_idx = text.find(keyword_processed, start_pos)
            if start_idx == -1:
                break
            
            is_standalone = True # Basic check, can be refined
            if start_idx > 0 and text[start_idx-1].isalnum():
                is_standalone = False
            if start_idx + len(keyword_processed) < len(text) and text[start_idx + len(keyword_processed)].isalnum():
                is_standalone = False
            
            if is_standalone:
                end_idx = start_idx + len(keyword_processed)
                spans.append((start_idx, end_idx, aspect_name))
                logger.debug(f"Keyword match: '{keyword_processed}' for aspect '{aspect_name}' at {start_idx}-{end_idx}")
            
            start_pos = start_idx + len(keyword_processed)
    
    return spans

# Data augmentation functions
def back_translate_text(text, translator, intermediate_lang='en', max_retries=2):
    """
    Perform back-translation: Greek -> Intermediate Language -> Greek
    Optimized for speed with reduced retries and delays.
    """
    if not TRANSLATION_AVAILABLE:
        logger.warning("Translation not available, returning original text")
        return text
    
    if not text or len(text.strip()) < 3:
        return text
    
    for attempt in range(max_retries):
        try:
            # Reduced delay for speed
            time.sleep(0.05)  # Reduced from 0.1
            
            # Translate to intermediate language
            intermediate = translator.translate(text, src='el', dest=intermediate_lang)
            if not intermediate or not intermediate.text:
                continue
                
            # Translate back to Greek
            time.sleep(0.05)  # Reduced from 0.1
            back_translated = translator.translate(intermediate.text, src=intermediate_lang, dest='el')
            if not back_translated or not back_translated.text:
                continue
                
            return back_translated.text
            
        except Exception as e:
            logger.debug(f"Translation attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(0.5)  # Reduced from 1 second
            continue
    
    # If all attempts fail, return original text
    logger.warning(f"All translation attempts failed for text: {text[:50]}...")
    return text

def augment_data_entries(data_entries, target_sentiment_ids, target_multiplier=3, fast_mode=False):
    """
    Augment data entries containing specified sentiment IDs using back-translation.
    
    Args:
        data_entries: List of data entries
        target_sentiment_ids: List of sentiment IDs to augment (0=negative, 1=neutral, 2=positive)
        target_multiplier: How many times to multiply the data (3 means original + 2 augmented = 3x total)
        fast_mode: If True, use simpler/faster augmentation methods
    
    Returns:
        List of augmented data entries
    """
    if not TRANSLATION_AVAILABLE or fast_mode:
        logger.warning("Using fast mode - simple augmentation instead of translation")
        return simple_augment_data_entries(data_entries, target_sentiment_ids, target_multiplier)
    
    logger.info(f"Starting data augmentation for sentiment IDs: {target_sentiment_ids}")
    logger.info(f"Target multiplier: {target_multiplier}x")
    
    # Initialize translator
    try:
        translator = Translator()
        # Quick test
        test_translation = translator.translate("test", src='en', dest='el')
        logger.info("Translator initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize translator: {e}")
        logger.warning("Falling back to simple augmentation")
        return simple_augment_data_entries(data_entries, target_sentiment_ids, target_multiplier)
    
    # Reduced languages for speed - only use English and French
    intermediate_languages = ['en', 'fr']  # Reduced from 5 to 2 languages
    
    # Find entries that need augmentation
    entries_to_augment = []
    for entry in data_entries:
        for aspect in entry.get('aspects_present', []):
            if aspect['sentiment_id'] in target_sentiment_ids:
                entries_to_augment.append(entry)
                break  # Only add entry once even if multiple aspects match
    
    original_count = len(entries_to_augment)
    target_count = original_count * target_multiplier
    augmentations_needed = target_count - original_count
    
    logger.info(f"Found {original_count} entries to augment")
    logger.info(f"Need to create {augmentations_needed} additional entries")
    
    if augmentations_needed <= 0:
        return data_entries
    
    # Create augmented entries
    augmented_entries = []
    
    # Calculate how many times to repeat the original entries
    repetitions_needed = augmentations_needed // original_count + 1
    
    translation_failures = 0
    
    # Estimate time and warn user
    estimated_time_seconds = augmentations_needed * 0.5  # ~0.5 seconds per entry with optimizations
    estimated_time_minutes = estimated_time_seconds / 60
    logger.info(f"Estimated completion time: {estimated_time_minutes:.1f} minutes")
    
    with tqdm(total=augmentations_needed, desc="Augmenting data") as pbar:
        for rep in range(repetitions_needed):
            if len(augmented_entries) >= augmentations_needed:
                break
                
            # Shuffle the order and pick different intermediate languages
            shuffled_entries = random.sample(entries_to_augment, len(entries_to_augment))
            
            for i, entry in enumerate(shuffled_entries):
                if len(augmented_entries) >= augmentations_needed:
                    break
                
                # Select intermediate language (cycling between fewer options)
                intermediate_lang = intermediate_languages[i % len(intermediate_languages)]
                
                # Create augmented entry
                augmented_entry = entry.copy()
                
                # Augment the original text
                original_text = entry['text_original']
                
                # Try translation, fall back to simple augmentation if it fails
                try:
                    augmented_text = back_translate_text(original_text, translator, intermediate_lang)
                    
                    # If translation returned the same text or very similar, apply simple augmentation
                    if augmented_text == original_text or len(augmented_text.strip()) < 3:
                        augmented_text = simple_text_augmentation(original_text, method=i % 3)
                        
                except Exception as e:
                    logger.debug(f"Translation failed for entry {i}: {e}")
                    translation_failures += 1
                    augmented_text = simple_text_augmentation(original_text, method=i % 3)
                
                # Update the entry
                augmented_entry['text_original'] = augmented_text
                augmented_entry['text_processed'] = preprocess_text(augmented_text)
                
                # Re-tokenize
                tokens, _ = tokenize_text(augmented_entry['text_processed'])
                augmented_entry['tokens'] = tokens
                
                # Keep the same aspects and sentiments (labels remain unchanged)
                augmented_entries.append(augmented_entry)
                pbar.update(1)
    
    if translation_failures > 0:
        logger.info(f"Translation failed for {translation_failures} entries, used simple augmentation instead")
    
    # Combine original and augmented data
    final_entries = data_entries + augmented_entries[:augmentations_needed]
    
    logger.info(f"Data augmentation completed. Total entries: {len(final_entries)} (was {len(data_entries)})")
    return final_entries


def simple_text_augmentation(text, method=0):
    """
    Simple text augmentation methods as fallback when translation is not available.
    """
    if not text or len(text.strip()) < 3:
        return text
    
    if method == 0:
        # Add minor punctuation variations
        return text + " ."
    elif method == 1:
        # Duplicate some words (very simple)
        words = text.split()
        if len(words) > 3:
            # Duplicate a random word
            dup_idx = random.randint(1, len(words) - 2)
            words.insert(dup_idx, words[dup_idx])
        return " ".join(words)
    else:
        # Change word order slightly (very simple)
        words = text.split()
        if len(words) > 2:
            # Swap two adjacent words
            swap_idx = random.randint(0, len(words) - 2)
            words[swap_idx], words[swap_idx + 1] = words[swap_idx + 1], words[swap_idx]
        return " ".join(words)


def simple_augment_data_entries(data_entries, target_sentiment_ids, target_multiplier=3):
    """
    Simple augmentation fallback when translation is not available.
    """
    logger.info(f"Using simple augmentation for sentiment IDs: {target_sentiment_ids}")
    
    # Find entries that need augmentation
    entries_to_augment = []
    for entry in data_entries:
        for aspect in entry.get('aspects_present', []):
            if aspect['sentiment_id'] in target_sentiment_ids:
                entries_to_augment.append(entry)
                break  # Only add entry once even if multiple aspects match
    
    original_count = len(entries_to_augment)
    target_count = original_count * target_multiplier
    augmentations_needed = target_count - original_count
    
    logger.info(f"Found {original_count} entries to augment")
    logger.info(f"Need to create {augmentations_needed} additional entries")
    
    if augmentations_needed <= 0:
        return data_entries
    
    # Create augmented entries
    augmented_entries = []
    
    # Calculate how many times to repeat the original entries
    repetitions_needed = augmentations_needed // original_count + 1
    
    with tqdm(total=augmentations_needed, desc="Simple augmentation") as pbar:
        for rep in range(repetitions_needed):
            if len(augmented_entries) >= augmentations_needed:
                break
                
            # Shuffle the order
            shuffled_entries = random.sample(entries_to_augment, len(entries_to_augment))
            
            for i, entry in enumerate(shuffled_entries):
                if len(augmented_entries) >= augmentations_needed:
                    break
                
                # Create augmented entry
                augmented_entry = entry.copy()
                
                # Apply simple augmentation
                original_text = entry['text_original']
                augmented_text = simple_text_augmentation(original_text, method=i % 3)
                
                # Update the entry
                augmented_entry['text_original'] = augmented_text
                augmented_entry['text_processed'] = preprocess_text(augmented_text)
                
                # Re-tokenize
                tokens, _ = tokenize_text(augmented_entry['text_processed'])
                augmented_entry['tokens'] = tokens
                
                # Keep the same aspects and sentiments (labels remain unchanged)
                augmented_entries.append(augmented_entry)
                pbar.update(1)
    
    # Combine original and augmented data
    final_entries = data_entries + augmented_entries[:augmentations_needed]
    
    logger.info(f"Simple augmentation completed. Total entries: {len(final_entries)} (was {len(data_entries)})")
    return final_entries

def process_data(input_file, output_dir_base, aspect_keywords_file, use_text_lemma=False, create_augmented=False, fast_augmentation=False):
    """
    Process data for BERT-based aspect detection and sentiment classification.
    Optionally create augmented datasets for improved sentiment classification.
    """
    with open(aspect_keywords_file, "r", encoding="utf-8") as f:
        aspect_keywords_map = json.load(f)
    
    df = pd.read_csv(input_file)
    
    # Determine output directory based on text column used
    output_dir = os.path.join(output_dir_base, "f_data_bert_lemma" if use_text_lemma else "f_data_bert")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory set to: {output_dir}")

    processed_data_all = []
    sentiment_to_id = {"negative": 0, "neutral": 1, "positive": 2}
    filtered_count = 0  # Track how many entries were filtered out
    
    text_column = "text_lemma" if use_text_lemma else "reviews"
    logger.info(f"Using '{text_column}' column for text content.")

    if 'type' not in df.columns:
        logger.error("The CSV file must contain a 'type' column for train/test splitting.")
        return

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing reviews ({text_column})"):
        if text_column not in row or pd.isna(row[text_column]):
            logger.warning(f"Row {idx} has no '{text_column}' value or it's NaN, skipping.")
            continue
        
        text_original = str(row[text_column])
        text_processed = preprocess_text(text_original)
        rated_aspects_info = extract_rated_aspects(row)

        char_spans_found = []
        detected_aspect_names_in_text = set()

        for aspect_obj in rated_aspects_info:
            aspect_name = aspect_obj["aspect"]
            # Use preprocessed aspect name for keywords if not in map, or use mapped keywords
            default_keywords = [preprocess_text(aspect_name)]
            keywords = aspect_keywords_map.get(aspect_name, default_keywords)
            
            # Important: find_aspect_terms_in_text expects preprocessed text
            current_aspect_spans = find_aspect_terms_in_text(text_processed, aspect_name, keywords)
            
            if current_aspect_spans:
                char_spans_found.extend(current_aspect_spans)
                detected_aspect_names_in_text.add(aspect_name) # Add original aspect name

        # Get tokens directly
        tokens, _ = tokenize_text(text_processed) # We don't need offsets in the final output
        
        # Filter rated_aspects to only include those found in the text
        # This is crucial for the aspect detection task
        final_aspects_for_output = [
            asp for asp in rated_aspects_info if asp["aspect"] in detected_aspect_names_in_text
        ]
        
        # If no aspects were mentioned OR detected in the text, this sample might be less useful for aspect *detection*
        # but could still be useful for sentiment if an aspect is given.
        # For now, we keep all, assuming the downstream model handles "no aspect detected".

        data_entry = {
            "text_original": text_original, # Keep original for reference if needed
            "text_processed": text_processed,
            "tokens": tokens,
            "aspects_present": [{ # Aspects detected in text (for aspect detection model)
                "aspect_category": asp["aspect"], # The category, e.g., "Μπαταρία"
                "sentiment_id": sentiment_to_id[asp["sentiment_str"]]
            } for asp in final_aspects_for_output],
        }
        
        # Only keep entries that have at least one aspect with sentiment
        # Empty aspects_present arrays are useless for both ATC and ASC training
        if data_entry["aspects_present"]:
            # Store the entry with type information for splitting
            processed_data_all.append((data_entry, row.get("type", "unknown")))
        else:
            filtered_count += 1

    logger.info(f"Total processed entries: {len(processed_data_all)}")
    logger.info(f"Filtered out entries: {filtered_count}")

    # Splitting data based on 'type' column
    train_data = [d[0] for d in processed_data_all if d[1] == 'train']
    test_val_data = [d[0] for d in processed_data_all if d[1] == 'test']
    
    if not test_val_data:
        logger.warning("No data marked as 'test' found. Val/Test sets will be empty.")
        val_data, test_data = [], []
    else:
        val_data, test_data = train_test_split(test_val_data, test_size=0.5, random_state=1312) # 50/50 split for val/test

    logger.info(f"Train data size: {len(train_data)}")
    logger.info(f"Validation data size: {len(val_data)}")
    logger.info(f"Test data size: {len(test_data)}")

    # Save datasets
    full_file_path = os.path.join(output_dir, "all_data.json")
    train_file_path = os.path.join(output_dir, "train_data.json")
    val_file_path = os.path.join(output_dir, "val_data.json")
    test_file_path = os.path.join(output_dir, "test_data.json")

    def save_to_jsonl(data, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        logger.info(f"Saved data to {file_path}")

    save_to_jsonl([d[0] for d in processed_data_all], full_file_path)
    save_to_jsonl(train_data, train_file_path)
    save_to_jsonl(val_data, val_file_path)
    save_to_jsonl(test_data, test_file_path)

    # Create augmented datasets if requested
    if create_augmented:
        create_augmented_datasets(output_dir, train_data, val_data, test_data, [d[0] for d in processed_data_all], fast_augmentation)

    return full_file_path, train_file_path, val_file_path, test_file_path


def create_augmented_datasets(base_output_dir, train_data, val_data, test_data, all_data, fast_mode=False):
    """
    Create augmented datasets specifically for ASC training.
    Augments neutral (sentiment_id=1) and negative (sentiment_id=0) sentiments to 3x their original size.
    """
    logger.info("Creating augmented datasets for improved ASC training...")
    
    if fast_mode:
        logger.info("Using FAST MODE - simple augmentation instead of translation")
    
    # Create augmented directory
    augmented_dir = base_output_dir + "_augmented"
    os.makedirs(augmented_dir, exist_ok=True)
    logger.info(f"Augmented directory set to: {augmented_dir}")
    
    # Target sentiment IDs to augment: 0=negative, 1=neutral
    target_sentiment_ids = [0, 1]  # negative and neutral
    target_multiplier = 3  # 3x the original size
    
    # Augment each dataset separately
    logger.info("Augmenting training data...")
    augmented_train_data = augment_data_entries(train_data, target_sentiment_ids, target_multiplier, fast_mode)
    
    logger.info("Augmenting validation data...")
    augmented_val_data = augment_data_entries(val_data, target_sentiment_ids, target_multiplier, fast_mode)
    
    logger.info("Augmenting test data...")
    augmented_test_data = augment_data_entries(test_data, target_sentiment_ids, target_multiplier, fast_mode)
    
    logger.info("Augmenting all data...")
    augmented_all_data = augment_data_entries(all_data, target_sentiment_ids, target_multiplier, fast_mode)
    
    # Save augmented datasets
    def save_to_jsonl(data, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        logger.info(f"Saved augmented data to {file_path}")
    
    augmented_full_file_path = os.path.join(augmented_dir, "all_data.json")
    augmented_train_file_path = os.path.join(augmented_dir, "train_data.json")
    augmented_val_file_path = os.path.join(augmented_dir, "val_data.json")
    augmented_test_file_path = os.path.join(augmented_dir, "test_data.json")
    
    save_to_jsonl(augmented_all_data, augmented_full_file_path)
    save_to_jsonl(augmented_train_data, augmented_train_file_path)
    save_to_jsonl(augmented_val_data, augmented_val_file_path)
    save_to_jsonl(augmented_test_data, augmented_test_file_path)
    
    # Log final statistics
    logger.info(f"Augmented dataset statistics:")
    logger.info(f"  All data: {len(all_data)} -> {len(augmented_all_data)} (+{len(augmented_all_data) - len(all_data)})")
    logger.info(f"  Train: {len(train_data)} -> {len(augmented_train_data)} (+{len(augmented_train_data) - len(train_data)})")
    logger.info(f"  Validation: {len(val_data)} -> {len(augmented_val_data)} (+{len(augmented_val_data) - len(val_data)})")
    logger.info(f"  Test: {len(test_data)} -> {len(augmented_test_data)} (+{len(augmented_test_data) - len(test_data)})")
    
    return augmented_full_file_path, augmented_train_file_path, augmented_val_file_path, augmented_test_file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for BERT-based aspect analysis.")
    parser.add_argument('--input_file', default='data/reviews_NoDuplicates_TrainTest.csv',
                        help='Path to input CSV file (must contain a "type" column).')
    parser.add_argument('--output_dir_base', default='data',
                        help='Base directory to save processed data folders (f_data_bert and f_data_bert_lemma).')
    parser.add_argument('--create_augmented', action='store_true',
                        help='Create augmented datasets for improved ASC training by augmenting neutral and negative sentiments to 3x their original size using back-translation.')
    parser.add_argument('--fast_augmentation', action='store_true',
                        help='Use fast augmentation mode (simple text modifications instead of translation). Much faster but potentially less effective.')
    
    args = parser.parse_args()

    # Process for 'reviews' column
    logger.info("Processing data using 'reviews' column...")
    process_data(
        input_file=args.input_file,
        output_dir_base=args.output_dir_base,
        aspect_keywords_file='data/aspect_keywords_map.json', # Hardcoded path
        use_text_lemma=False,
        create_augmented=args.create_augmented,
        fast_augmentation=args.fast_augmentation
    )

    # Process for 'text_lemma' column
    logger.info("Processing data using 'text_lemma' column...")
    process_data(
        input_file=args.input_file,
        output_dir_base=args.output_dir_base,
        aspect_keywords_file='data/aspect_keywords_lemma.json', # Hardcoded path
        use_text_lemma=True,
        create_augmented=args.create_augmented,
        fast_augmentation=args.fast_augmentation
    )
    
    if args.create_augmented:
        if args.fast_augmentation:
            logger.info("Data preparation finished for both 'reviews' and 'text_lemma' columns with FAST augmentation.")
        else:
            logger.info("Data preparation finished for both 'reviews' and 'text_lemma' columns with translation-based augmentation.")
    else:
        logger.info("Data preparation finished for both 'reviews' and 'text_lemma' columns.")
