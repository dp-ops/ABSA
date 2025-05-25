import json
import random

# ANSI escape code for green color
GREEN = '\033[92m'
RESET = '\033[0m'

def load_data(file_path):
    """Loads data from a JSON file where each line is a JSON object."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping line due to error: {e} in line: {line.strip()}")
    return data

def display_highlighted_text(item):
    """Displays the text with BIO-tagged aspects highlighted in green."""
    text = item.get('text')
    tokens = item.get('tokens')
    bio_labels = item.get('bio_labels')

    if not text or not tokens or not bio_labels:
        print("Skipping item due to missing 'text', 'tokens', or 'bio_labels' field.")
        return

    # Remove [CLS] and [SEP] tokens and their corresponding labels if present
    if tokens[0] == "[CLS]":
        tokens = tokens[1:]
        bio_labels = bio_labels[1:]
    if tokens and tokens[-1] == "[SEP]":
        tokens = tokens[:-1]
        bio_labels = bio_labels[:-1]
    
    if len(tokens) != len(bio_labels):
        print(f"Skipping item due to mismatched tokens and bio_labels length for text: {text}")
        print(f"Tokens ({len(tokens)}): {tokens}")
        print(f"BIO Labels ({len(bio_labels)}): {bio_labels}")
        return

    highlighted_text_parts = []
    original_text_cursor = 0

    # Reconstruct text from tokens and apply highlighting
    # This is a simplified reconstruction. A more robust method might be needed
    # if tokenization significantly alters word boundaries or introduces many ## prefixes.
    
    current_word = ""
    for i, token in enumerate(tokens):
        label = bio_labels[i]
        
        token_to_append = token
        if token.startswith("##"):
            token_to_append = token[2:]
        else:
            if current_word: # Add space before new word if it's not a sub-token
                 highlighted_text_parts.append(" ")
            
        if label.startswith("B-ASP") or label.startswith("I-ASP"):
            highlighted_text_parts.append(GREEN + token_to_append + RESET)
        else:
            highlighted_text_parts.append(token_to_append)
        
        current_word += token_to_append


    print(f"Original Text: {text}")
    print(f"Highlighted:   {''.join(highlighted_text_parts)}")
    print("-" * 30)


if __name__ == "__main__":
    file_path = "data/filtered_data_xlm/processed_aspect_data_test.json"
    all_data = load_data(file_path)

    if not all_data:
        print(f"No data loaded from {file_path}. Exiting.")
    else:
        if len(all_data) < 20:
            print(f"Warning: Fewer than 20 items in the dataset. Displaying all {len(all_data)} items.")
            sample_data = all_data
        else:
            sample_data = random.sample(all_data, 20)

        print(f"Displaying 20 random samples with BIO tagging from {file_path}:\n")
        for item in sample_data:
            display_highlighted_text(item) 