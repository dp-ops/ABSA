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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the tokenizer for RoBERTa model
tokenizer = AutoTokenizer.from_pretrained("pchatz/palobert-base-greek-social-media-v2")

# Preprocessing function for RoBERTa model
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

# Tokenization function
def tokenize_text(text):
    """Tokenize the preprocessed text and return tokens with offsets."""
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
            # Convert to sentiment values
            sentiment_str = str(row[col_name]).strip()
            sentiment_val = -1 if sentiment_str == '-1' else int(sentiment_str)
            sentiment_str = "positive" if sentiment_val == 1 else "negative" if sentiment_val == -1 else "neutral"
            
            extracted.append({
                "aspect": col_name,
                "sentiment_val": sentiment_val,
                "sentiment_str": sentiment_str
            })
    return extracted

def find_aspect_terms_in_text(text, aspect_name, keywords):
    """Find aspect terms in text using improved matching."""
    # Preprocess both text and keywords using the same function
    text_processed = preprocess_text(text)
    spans = []
    
    # Process keywords - remove diacritics and convert to lowercase
    processed_keywords = []
    for keyword in keywords:
        processed_keyword = preprocess_text(keyword)
        if processed_keyword and len(processed_keyword) >= 3:  # Ensure valid keywords
            processed_keywords.append(processed_keyword)
    
    # First try exact matches of the aspect name
    aspect_name_processed = preprocess_text(aspect_name)
    start_idx = text_processed.find(aspect_name_processed)
    if start_idx != -1:
        end_idx = start_idx + len(aspect_name_processed)
        spans.append((start_idx, end_idx, aspect_name))
        logger.debug(f"Direct aspect name match: '{aspect_name_processed}' at {start_idx}-{end_idx}")
    
    # Then try keyword matching
    for keyword_processed in processed_keywords:
        if not keyword_processed or len(keyword_processed) < 3:  # Skip empty or very short keywords
            continue
            
        start_pos = 0
        while True:
            start_idx = text_processed.find(keyword_processed, start_pos)
            if start_idx == -1:
                break
                
            # Check if it's a standalone word
            is_standalone = True
            if start_idx > 0 and text_processed[start_idx-1].isalnum():
                is_standalone = False
            if start_idx + len(keyword_processed) < len(text_processed) and text_processed[start_idx + len(keyword_processed)].isalnum():
                is_standalone = False
                
            if is_standalone:
                end_idx = start_idx + len(keyword_processed)
                spans.append((start_idx, end_idx, aspect_name))
                logger.debug(f"Keyword match: '{keyword_processed}' for aspect '{aspect_name}' at {start_idx}-{end_idx}")
            
            # Move to position after this match
            start_pos = start_idx + len(keyword_processed)
    
    return spans

def generate_bio_labels(text, char_spans):
    """Generate BIO labels for tokens based on character spans."""
    # Preprocess text for RoBERTa
    processed_text = preprocess_text(text)
    tokens, offsets = tokenize_text(processed_text)
    labels = ["O"] * len(tokens)  # Initialize all labels as 'O'
    
    # Sort spans by start position to handle overlapping spans correctly
    char_spans.sort(key=lambda x: x[0])
    
    labeled_tokens = []
    
    for (span_start, span_end, aspect_name) in char_spans:
        # Track whether we're inside an aspect span
        inside_aspect = False
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == 0 and tok_end == 0:  # Skip special tokens
                continue
            # Check if token overlaps with the aspect span
            if tok_start < span_end and tok_end > span_start:
                if not inside_aspect:
                    labels[i] = "B-ASP"  # First token of the aspect
                    inside_aspect = True
                    labeled_tokens.append(tokens[i])
                else:
                    labels[i] = "I-ASP"  # Subsequent tokens of the aspect
                    labeled_tokens.append(tokens[i])
            elif inside_aspect:  # If we were inside but now we're not
                inside_aspect = False
    
    # Debug information
    if labeled_tokens:
        logger.debug(f"Text: {processed_text}")
        logger.debug(f"Found aspects: {', '.join([s[2] for s in char_spans])}")
        logger.debug(f"Labeled tokens: {', '.join(labeled_tokens)}")
    
    return tokens, labels

def process_data(input_file, output_dir, aspect_keywords_file, filter_noaspects=False, use_reviews_column=False):
    """
    Process data with improved BIO tagging for RoBERTa model.
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file
    output_dir : str
        Directory to save processed data
    aspect_keywords_file : str
        Path to aspect keywords mapping JSON file
    filter_noaspects : bool
        If True, filter out examples without detected aspect terms
    use_reviews_column : bool
        If True, use the 'reviews' column for text content instead of 'text_proc'
    """
    # Load aspect keywords from JSON
    with open(aspect_keywords_file, "r", encoding="utf-8") as f:
        aspect_keywords_map = json.load(f)
    
    # Load CSV data
    df = pd.read_csv(input_file)
    
    # Process each review
    processed_data = []
    sentiment_to_id = {"negative": 0, "neutral": 1, "positive": 2}
    
    total_rows = len(df)
    rows_with_aspects = 0
    total_aspects_found = 0
    
    # Determine which column to use for text content
    text_column = "reviews" if use_reviews_column else "text_proc"
    logger.info(f"Using '{text_column}' column for text content")
    
    for idx, row in tqdm(df.iterrows(), total=total_rows, desc="Processing reviews"):
        # Check if the chosen column exists in the data
        if text_column not in row or pd.isna(row[text_column]):
            logger.warning(f"Row {idx} has no '{text_column}' value, skipping.")
            continue
            
        # Apply RoBERTa preprocessing to text
        text = preprocess_text(row[text_column])
        rated_aspects = extract_rated_aspects(row)
        
        # Find aspect spans
        char_spans = []
        
        # For each rated aspect, try to find it in the text
        for aspect_obj in rated_aspects:
            aspect_name = aspect_obj["aspect"]
            keywords = aspect_keywords_map.get(aspect_name, [aspect_name.lower()])
            
            # Find all occurrences of keywords in text
            aspect_spans = find_aspect_terms_in_text(text, aspect_name, keywords)
            char_spans.extend(aspect_spans)
        
        # Generate BIO labels
        tokens, bio_labels = generate_bio_labels(text, char_spans)
        
        # Track metrics
        has_aspects = any(label != "O" for label in bio_labels)
        if has_aspects:
            rows_with_aspects += 1
            total_aspects_found += len([l for l in bio_labels if l.startswith("B-")])
        
        # Debug: Check if any aspects were identified
        if not has_aspects and rated_aspects:
            logger.debug(f"No aspect spans found in text: {text} for aspects: {[a['aspect'] for a in rated_aspects]}")
        
        # Skip entries without aspects if filter is enabled
        if filter_noaspects and not has_aspects:
            continue
        
        # Prepare output
        processed_data.append({
            "text": text,
            "tokens": tokens,
            "bio_labels": bio_labels,
            "aspects": [{
                "aspect": asp["aspect"],
                "sentiment_id": sentiment_to_id[asp["sentiment_str"]]
            } for asp in rated_aspects],
        })
    
    logger.info(f"Processed {total_rows} rows")
    logger.info(f"Found aspects in {rows_with_aspects} rows ({rows_with_aspects/total_rows*100:.2f}%)")
    logger.info(f"Total number of aspects found: {total_aspects_found}")
    logger.info(f"Final dataset size after filtering: {len(processed_data)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full dataset
    full_file = os.path.join(output_dir, "processed_aspect_data.json")
    with open(full_file, 'w', encoding='utf-8') as f:
        for entry in processed_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Split into train/val/test datasets
    train_data, temp_data = train_test_split(processed_data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    # Save train data
    train_file = os.path.join(output_dir, "processed_aspect_data_train.json")
    with open(train_file, 'w', encoding='utf-8') as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Save validation data
    val_file = os.path.join(output_dir, "processed_aspect_data_val.json")
    with open(val_file, 'w', encoding='utf-8') as f:
        for entry in val_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Save test data
    test_file = os.path.join(output_dir, "processed_aspect_data_test.json")
    with open(test_file, 'w', encoding='utf-8') as f:
        for entry in test_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(
        f"Processed data saved to:\n"
        f"- Full dataset: {full_file}\n"
        f"- Train: {train_file}\n"
        f"- Validation: {val_file}\n"
        f"- Test: {test_file}"
    )
    
    return full_file, train_file, val_file, test_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data for aspect-based sentiment analysis with RoBERTa model")
    parser.add_argument('--input_file', default='data/test_2731_reviews.csv',
                        help='Path to input CSV file')
    parser.add_argument('--output_dir', default='data/filtered_data_r',
                        help='Directory to save processed data')
    parser.add_argument('--aspect_keywords', default='data/aspect_keywords_map.json',
                        help='Path to aspect keywords mapping')
    parser.add_argument('--filter_noaspects', action='store_true',
                        help='Filter out examples without detected aspect terms')
    parser.add_argument('--use_reviews_column', action='store_true',
                        help='Use the "reviews" column instead of "text_proc" for text content')
    
    args = parser.parse_args()
    
    # Override output_dir if use_reviews_column is selected
    if args.use_reviews_column:
        args.output_dir = 'data/filtered_review_data_r'
    
    process_data(
        args.input_file, 
        args.output_dir, 
        args.aspect_keywords, 
        args.filter_noaspects,
        args.use_reviews_column
    ) 