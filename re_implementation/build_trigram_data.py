import logging
import pickle
import time
import itertools
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import multiprocessing
import os
from collections import Counter

# --- Configuration ---
PROCESSED_DATA_DIR = Path("processed_corpus_data")
SIMPLE_TOKENIZED_CORPUS_FILE = PROCESSED_DATA_DIR / "simple_jawiki_tokenized.txt"
TRIGRAM_COUNTS_FILE = PROCESSED_DATA_DIR / "jp_trigram_counts.pkl"
NUM_PROCESSES = os.cpu_count()
CHUNK_SIZE = 100_000 # Number of lines per chunk
MIN_TRIGRAM_COUNT = 5 # Only keep trigrams that appear 5 or more times

def setup_logging():
    log_dir = Path("logs/filtering")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler()]
    )
    logging.info("Logging configured for trigram data build.")

def process_chunk_for_counts(chunk_of_lines):
    """
    Worker function: processes a chunk of lines and returns a local Counter of trigrams.
    """
    local_trigram_counts = Counter()
    for sentence in chunk_of_lines:
        tokens = sentence.strip().split()
        if len(tokens) >= 3:
            for i in range(len(tokens) - 2):
                trigram = (tokens[i], tokens[i+1], tokens[i+2])
                local_trigram_counts[trigram] += 1
    return local_trigram_counts

def read_in_chunks(file_path, chunk_size):
    """Generator to read a file in chunks of lines."""
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            chunk = list(itertools.islice(f, chunk_size))
            if not chunk:
                break
            yield chunk

def run_trigram_build_parallel():
    """
    Reads the tokenized corpus, builds a Counter of all trigrams, and prunes
    it based on a minimum count, using multiple processes.
    """
    logger = logging.getLogger('TrigramBuilder')
    logger.info("====== Starting Trigram Counts Construction (Parallel Version) ======")

    if not SIMPLE_TOKENIZED_CORPUS_FILE.exists():
        logger.error(f"Input file not found: {SIMPLE_TOKENIZED_CORPUS_FILE}")
        return

    if TRIGRAM_COUNTS_FILE.exists():
        logger.warning(f"Output file already exists: {TRIGRAM_COUNTS_FILE}. Aborting.")
        return

    start_time = time.time()
    
    try:
        with open(SIMPLE_TOKENIZED_CORPUS_FILE, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f)
    except Exception as e:
        logger.error(f"Could not read the input file to determine its size: {e}")
        return
        
    final_trigram_counts = Counter()

    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        logger.info(f"Starting worker pool with {NUM_PROCESSES} processes.")
        chunk_generator = read_in_chunks(SIMPLE_TOKENIZED_CORPUS_FILE, CHUNK_SIZE)
        total_chunks = (total_lines + CHUNK_SIZE - 1) // CHUNK_SIZE
        logger.info(f"Distributing {total_chunks} chunks to workers...")
        
        pbar = tqdm(pool.imap_unordered(process_chunk_for_counts, chunk_generator), total=total_chunks, desc="Processing Chunks")
        
        for local_counts in pbar:
            final_trigram_counts.update(local_counts)

    total_unique_trigrams = len(final_trigram_counts)
    logger.info(f"All chunks processed. Found {total_unique_trigrams:,} unique trigrams before pruning.")
    
    # --- Pruning Step ---
    logger.info(f"Pruning trigrams with count < {MIN_TRIGRAM_COUNT}...")
    pruned_trigrams = {trigram for trigram, count in final_trigram_counts.items() if count >= MIN_TRIGRAM_COUNT}
    logger.info(f"Pruned from {total_unique_trigrams:,} to {len(pruned_trigrams):,} trigrams.")
    
    # We only need the set of valid trigrams for filtering, not their counts.
    # Saving just the set is more efficient.
    logger.info(f"Saving pruned trigram set to {TRIGRAM_COUNTS_FILE} using pickle...")
    try:
        with open(TRIGRAM_COUNTS_FILE, 'wb') as f_out:
            pickle.dump(pruned_trigrams, f_out)
        logger.info(f"✅ Successfully saved pruned trigram data.")
    except Exception as e:
        logger.error(f"Failed to save trigram data: {e}", exc_info=True)

    duration = time.time() - start_time
    logger.info(f"Trigram count and prune construction finished in {duration / 60:.2f} minutes.")
    logger.info("====== Trigram Counts Construction Finished ======")

if __name__ == '__main__':
    setup_logging()
    run_trigram_build_parallel()