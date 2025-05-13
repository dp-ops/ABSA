import json
import os
import pandas as pd
from transformers import AutoTokenizer

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")

# Load aspect keywords from JSON
with open("data/aspect_keywords_map.json", "r", encoding="utf-8") as f:
    aspect_keywords_map = json.load(f)

# Simplified preprocessing (text_proc is already cleaned)
def preprocess_text(text):
    return str(text)  # Just ensure it's a string

# Tokenization function
def tokenize_text(text):
    """Tokenize the preprocessed text and return tokens with offsets."""
    encoded = tokenizer(text, return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    offsets = encoded["offset_mapping"]
    return tokens, offsets

# Function to extract aspects and sentiments from CSV row
def extract_rated_aspects(row):
    aspect_columns = [
        "Ποιότητα κλήσης", "Φωτογραφίες", "Καταγραφή Video", "Ταχύτητα",
        "Ανάλυση οθόνης", "Μπαταρία", "Σχέση ποιότητας τιμής", "Μουσική"
    ]
    extracted = []
    for col_name in aspect_columns:
        sentiment_val = row[col_name]
        if pd.notna(sentiment_val) and str(sentiment_val).strip() in ['-1', '0', '1', '-']:  # Handle hyphen case
            # Convert hyphen to -1 (assuming '-' means negative)
            sentiment_str = str(sentiment_val).strip()
            if sentiment_str == '-':
                sentiment_val = -1
            else:
                sentiment_val = int(sentiment_val)
            
            sentiment_str = "positive" if sentiment_val == 1 else "negative" if sentiment_val == -1 else "neutral"
            extracted.append({
                "aspect": col_name,
                "sentiment_val": sentiment_val,
                "sentiment_str": sentiment_str
            })
    return extracted

# Function to generate BIO labels
def generate_bio_labels(text, char_spans):
    tokens, offsets = tokenize_text(text)
    labels = ["O"] * len(tokens)  # Initialize all labels as 'O'
    
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
                else:
                    labels[i] = "I-ASP"  # Subsequent tokens of the aspect
            else:
                inside_aspect = False
    return tokens, labels  # Return both tokens and labels

# Main processing function
def process_data(input_file, output_dir):
    df = pd.read_csv(input_file)
    processed_data = []
    sentiment_to_id = {"negative": 0, "neutral": 1, "positive": 2}

    for idx, row in df.iterrows():
        text = preprocess_text(row["text_proc"])  # Use text_proc instead of comment
        rated_aspects = extract_rated_aspects(row)
        
        # Find aspect spans using keywords
        char_spans = []
        for aspect_obj in rated_aspects:
            aspect_name = aspect_obj["aspect"]
            keywords = aspect_keywords_map.get(aspect_name, [aspect_name.lower()])
            for keyword in keywords:
                start_idx = text.find(keyword)
                if start_idx != -1:
                    char_spans.append((start_idx, start_idx + len(keyword), aspect_name))
        
        # Generate BIO labels
        tokens, bio_labels = generate_bio_labels(text, char_spans)
        
        # Prepare output
        processed_data.append({
            "text": text,
            "tokens": tokens,
            "bio_labels": bio_labels,
            "aspects": [{
                "aspect": asp["aspect"],
                "sentiment": asp["sentiment_str"],
                "sentiment_id": sentiment_to_id[asp["sentiment_str"]]
            } 
            # for extra data as to possitive or negative general sentiment and star rating, coomment if not needed
            for asp in rated_aspects],
            "overall_sentiment": {
                "sent_predicted": row.get("sentiment_predicted", None),
                "sent_star": row.get("sentiment_star", None)
            }
        })
    
    # Save output
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "processed_aspect_data.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"Processed data saved to {output_file}")

# Example usage
if __name__ == "__main__":
    input_file = "data/test_2731_reviews.csv"
    output_dir = "data/processed_data"
    process_data(input_file, output_dir)