import logging
import json
import time
import itertools
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
import os
import tempfile
import pickle

# --- Configuration ---
PROCESSED_DATA_DIR = Path("processed_corpus_data")
SIMPLE_TOKENIZED_CORPUS_FILE = PROCESSED_DATA_DIR / "simple_jawiki_tokenized.txt"
COOCCURRENCE_FILE = PROCESSED_DATA_DIR / "jp_cooccurrence_counts.json"
NUM_PROCESSES = os.cpu_count()
CHUNK_SIZE = 100_000
MIN_COOCCURRENCE_COUNT = 5

def setup_logging():
    log_dir = Path("logs/cooccurrence_calculation")
    log_dir.mkdir(exist_ok=True)
    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    file_handler = logging.FileHandler(
        log_dir / f"cooccurrence_calculation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        encoding='utf-8'
    )
    file_handler.setFormatter(log_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    logging.info("Logging configured for co-occurrence calculation.")

def create_cooccurrence_visualizations_de(cooccurrence_counts, output_dir="figures", top_n=25):
    logger = logging.getLogger('CooccurrenceVisualizer')
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    logger.info(f"Generating co-occurrence visualizations in {output_path.resolve()}...")

    if not cooccurrence_counts:
        logger.warning("No co-occurrence data available. Skipping visualization.")
        return

    try:
        plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'MS Gothic', 'Noto Sans CJK JP', 'sans-serif']
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception as e:
        logger.warning(f"Could not set a specific Japanese font list for plots: {e}")

    fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.45)))
    top_items = cooccurrence_counts.most_common(top_n)
    
    if top_items:
        labels = [f"'{item[0][0]}' & '{item[0][1]}'" for item in top_items]
        counts = [item[1] for item in top_items]
        sns.barplot(x=counts, y=labels, palette="magma", hue=labels, ax=ax, legend=False)
        
        ax.set_title(f'Top {len(labels)} am häufigsten gemeinsam auftretende Wortpaare', fontsize=16)
        ax.set_xlabel('Kookkurrenz-Häufigkeit (Anzahl der Sätze)', fontsize=12)
        ax.set_ylabel('Wortpaar', fontsize=12)
        
        plt.tight_layout()
        plot_file_path = output_path / "top_n_kookkurrenz_haeufigkeit.png"
        try:
            fig.savefig(plot_file_path, dpi=300)
            logger.info(f"📊 German Top N co-occurrence plot saved to: {plot_file_path}")
        except Exception as e:
            logger.error(f"Failed to save co-occurrence plot: {e}", exc_info=True)
    
    plt.close(fig)

def process_chunk_and_save(args):
    """
    Worker function: processes a chunk of lines, creates a Counter,
    and saves it to a temporary file using pickle.
    Returns the path to the temporary file.
    """
    chunk_of_lines, temp_dir_path, chunk_index = args
    local_counts = Counter()
    for sentence in chunk_of_lines:
        tokens = sentence.strip().split()
        unique_tokens = set(tokens)
        
        if len(unique_tokens) >= 2:
            for token1, token2 in itertools.combinations(sorted(list(unique_tokens)), 2):
                local_counts[(token1, token2)] += 1
    
    # Save the partial counter to a unique temporary file
    temp_file_path = temp_dir_path / f"partial_counts_{chunk_index}.pkl"
    with open(temp_file_path, 'wb') as f_out:
        pickle.dump(local_counts, f_out)
        
    return temp_file_path

def read_in_chunks(file_path, chunk_size):
    """Generator to read a file in chunks."""
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            chunk = list(itertools.islice(f, chunk_size))
            if not chunk:
                break
            yield chunk

def run_cooccurrence_calculation():
    """
    Calculates word co-occurrence statistics using multiple processes with a
    disk-based intermediate storage to prevent memory issues and BrokenPipeErrors.
    """
    logger = logging.getLogger('CooccurrenceCalculator')
    logger.info("====== Starting Co-occurrence Calculation Script (Parallel Version) ======")

    if not SIMPLE_TOKENIZED_CORPUS_FILE.exists():
        logger.error(f"Input file not found: {SIMPLE_TOKENIZED_CORPUS_FILE}")
        return

    if COOCCURRENCE_FILE.exists():
        logger.warning(f"Output file already exists: {COOCCURRENCE_FILE}. Aborting.")
        return

    start_time = time.time()
    
    try:
        with open(SIMPLE_TOKENIZED_CORPUS_FILE, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f)
        logger.info(f"Corpus contains {total_lines:,} sentences.")
    except Exception as e:
        logger.error(f"Could not read the input file to determine its size: {e}")
        return

    # Use a temporary directory that will be automatically cleaned up
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        logger.info(f"Using temporary directory for partial results: {temp_dir_path}")

        # --- Map Step: Process chunks and save results to disk ---
        with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
            logger.info(f"Starting worker pool with {NUM_PROCESSES} processes.")
            
            chunk_generator = read_in_chunks(SIMPLE_TOKENIZED_CORPUS_FILE, CHUNK_SIZE)
            
            # Prepare arguments for each worker: (chunk, temp_dir_path, chunk_index)
            # tasks = [(chunk, temp_dir_path, i) for i, chunk in enumerate(chunk_generator)]
            tasks_generator = ((chunk, temp_dir_path, i) for i, chunk in enumerate(chunk_generator))
            total_tasks = (total_lines + CHUNK_SIZE - 1) // CHUNK_SIZE

            logger.info(f"Distributing {total_tasks} chunks to workers...")
            
            # We don't need the return values here, just to ensure all tasks complete.
            # pool.imap_unordered can be slightly faster as it doesn't preserve order.
            for _ in tqdm(pool.imap_unordered(process_chunk_and_save, tasks_generator), total=total_tasks, desc="Processing Chunks"):
                pass # The loop consumes the iterator, ensuring all tasks run.

        logger.info("All workers have finished. All partial results saved to disk.")
        
        # --- Reduce Step: Merge temporary files from disk ---
        logger.info("Merging partial results from temporary files...")
        final_cooccurrence_counts = Counter()
        temp_files = sorted(list(temp_dir_path.glob("*.pkl")))
        
        for temp_file in tqdm(temp_files, desc="Merging Files"):
            with open(temp_file, 'rb') as f_in:
                partial_counter = pickle.load(f_in)
                final_cooccurrence_counts.update(partial_counter)
    
    # The temporary directory and its contents are automatically deleted when the 'with' block exits.
    logger.info("Temporary files merged and directory cleaned up.")

    total_pairs = sum(final_cooccurrence_counts.values())
    logger.info(f"Found {len(final_cooccurrence_counts):,} unique co-occurring pairs before pruning.")
    logger.info(f"Total co-occurrence instances counted: {total_pairs:,}")

    # --- Pruning, Visualization, and Saving (same as before) ---
    logger.info(f"Pruning pairs with count < {MIN_COOCCURRENCE_COUNT}...")
    pruned_counts = Counter({pair: count for pair, count in final_cooccurrence_counts.items() if count >= MIN_COOCCURRENCE_COUNT})
    logger.info(f"Pruned from {len(final_cooccurrence_counts):,} to {len(pruned_counts):,} pairs.")

    # create_cooccurrence_visualizations_de(pruned_counts)

    logger.info("Serializing pruned data for JSON storage...")
    serializable_counts = {'|'.join(pair): count for pair, count in pruned_counts.items()}
    output_data = {'counts': serializable_counts, 'total_pairs': total_pairs}

    try:
        with open(COOCCURRENCE_FILE, 'w', encoding='utf-8') as f:
            json.dump(output_data, f)
        logger.info(f"✅ Successfully saved co-occurrence data to {COOCCURRENCE_FILE}")
    except Exception as e:
        logger.error(f"Failed to save co-occurrence data: {e}", exc_info=True)

    duration = time.time() - start_time
    logger.info(f"Co-occurrence calculation finished in {duration / 60:.2f} minutes.")
    logger.info("====== Co-occurrence Calculation Script Finished ======")

if __name__ == '__main__':
    setup_logging()
    run_cooccurrence_calculation()