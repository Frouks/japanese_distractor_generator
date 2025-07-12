import json
from pathlib import Path
from collections import Counter
import time

# --- Configuration ---
COOCCURRENCE_FILE = Path("processed_corpus_data") / "jp_cooccurrence_counts.json"

def load_data(file_path):
    """Loads the co-occurrence data and converts it to a Counter."""
    if not file_path.exists():
        print(f"❌ ERROR: File not found at {file_path}")
        return None

    print(f"Loading data from {file_path}... (This may take a moment for a large file)")
    start_time = time.time()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert the 'word1|word2' string keys back into ('word1', 'word2') tuple keys
        reformatted_counts = {tuple(key.split('|')): count for key, count in data['counts'].items()}
        # Create a Counter object for easy sorting and analysis
        cooc_counter = Counter(reformatted_counts)
        
        duration = time.time() - start_time
        print(f"✅ Data loaded successfully in {duration:.2f} seconds.")
        print(f"Found {len(cooc_counter):,} unique pairs.")
        return cooc_counter

    except Exception as e:
        print(f"❌ An error occurred during loading: {e}")
        return None

def show_top_n(counter, n=20):
    """Displays the N most common items."""
    if not counter:
        return
    print(f"\n--- Top {n} Most Frequent Co-occurrence Pairs ---")
    if not counter:
        print("No data available.")
        return
    
    top_items = counter.most_common(n)
    for (w1, w2), count in top_items:
        print(f"  ({count:8,}) : ('{w1}', '{w2}')")

def show_bottom_n(counter, n=20):
    """Displays the N least common items."""
    if not counter:
        return
    print(f"\n--- Bottom {n} Least Frequent Co-occurrence Pairs ---")
    if not counter:
        print("No data available.")
        return
        
    # .most_common() can take a negative argument to get the least common items
    bottom_items = counter.most_common()[:-n-1:-1]
    for (w1, w2), count in bottom_items:
        print(f"  ({count:8,}) : ('{w1}', '{w2}')")

def search_for_word(counter, search_word):
    """Finds and displays all pairs containing a specific word."""
    if not counter:
        return
    print(f"\n--- Searching for pairs containing the word: '{search_word}' ---")
    
    # This list comprehension will find all relevant pairs
    found_pairs = [
        (pair, count) for pair, count in counter.items() if search_word in pair
    ]
    
    if not found_pairs:
        print(f"No pairs found containing '{search_word}'.")
        return
        
    # Sort the found pairs by frequency (most frequent first)
    found_pairs.sort(key=lambda item: item[1], reverse=True)
    
    print(f"Found {len(found_pairs)} pairs. Showing top 50:")
    for (w1, w2), count in found_pairs[:50]:
        # Make the search word stand out a bit
        other_word = w2 if w1 == search_word else w1
        print(f"  ({count:8,}) : ('{search_word}', '{other_word}')")


if __name__ == "__main__":
    cooc_counter = load_data(COOCCURRENCE_FILE)

    if cooc_counter:
        # --- Example Usage ---
        
        # 1. Show the most frequent pairs
        show_top_n(cooc_counter, n=25)
        
        # 2. Show the least frequent pairs (those that just met the threshold)
        show_bottom_n(cooc_counter, n=25)
        
        # 3. Search for a specific word. 
        search_for_word(cooc_counter, "猫") # Cat
        
        search_for_word(cooc_counter, "東京") # Tokyo

        search_for_word(cooc_counter, "走る") # run

        search_for_word(cooc_counter, "走った") # ran