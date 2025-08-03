import itertools
import logging
import multiprocessing
import os
import pickle
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import spacy
from tqdm import tqdm

from wiki_corpus_reader.wiki_corpus_reader import WikiCorpusReader

# --- Configuration ---
WIKI_EXTRACTED_PATH = "/Volumes/T7/Bachelorthesis/jawiki_data/jawiki_extracted"
PROCESSED_DATA_DIR = Path("processed_corpus_data")

# --- Test Mode Configuration ---
# Set to True to run a quick test on a small sample of the data.
# Set to False to run the full build process on the entire corpus.
RUN_IN_TEST_MODE = False
NUM_TEST_SENTENCES = 1000

# --- Dynamic Filename and Performance Tuning ---
if RUN_IN_TEST_MODE:
    DEPENDENCY_FILE = PROCESSED_DATA_DIR / "jp_dependency_relations_SAMPLE.pkl"
    NUM_PROCESSES = os.cpu_count()
    CHUNK_SIZE = 100
    MIN_RELATION_COUNT = 1
else:
    DEPENDENCY_FILE = PROCESSED_DATA_DIR / "jp_dependency_relations.pkl"
    NUM_PROCESSES = max(1, os.cpu_count() // 2)
    CHUNK_SIZE = 5_000
    MIN_RELATION_COUNT = 3

# --- Filtering Rules for the Build Process ---
EXCLUDED_DEP_LABELS = {
    'case',  # Particle markers (が, を, に).
    'aux',  # Auxiliary verbs (ます, ない).
    'punct',  # Punctuation (。, 、).
    'mark',  # Subordinating markers (ので, なら).
}
INCLUDED_POS_TAGS = {
    'NOUN', 'PROPN',  # Nouns, Proper Nouns.
    'VERB',  # Verbs.
    'ADJ',  # Adjectives.
    'ADV',  # Adverbs.
}


def setup_logging():
    """Sets up a logging configuration for the entire build script."""
    log_dir = Path("logs/filtering")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / f"dependency_build_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                                encoding='utf-8')
        ]
    )
    logging.info("Logging configured for dependency data build.")


def process_sentence_chunk(sentences):
    """
    The main "worker" function. It takes a list of sentences, parses them,
    and returns a filtered set of all dependency relations found within them.
    """
    nlp = spacy.load("ja_ginza")
    local_relations_counter = Counter()
    # Generator method
    docs = nlp.pipe(sentences)

    # Doc is an analyzed sentence
    for doc in docs:
        # Iterates over each word in the sentence -> a word can only have one head
        for token in doc:
            if token.dep_ in EXCLUDED_DEP_LABELS:
                continue
            # Checks if the token is the ROOT of the sentence
            if token.head == token:
                continue
                
            if token.head.pos_ not in INCLUDED_POS_TAGS or token.pos_ not in INCLUDED_POS_TAGS:
                continue

            relation = (token.dep_, token.head.lemma_, token.lemma_)
            local_relations_counter[relation] += 1

    return local_relations_counter


def read_sentences_in_chunks(sentence_generator, chunk_size):
    """
    A generator function that takes a sentence iterator and yields lists of sentences (chunks).
    """
    while True:
        chunk = list(itertools.islice(sentence_generator, chunk_size))
        if not chunk:
            break
        yield chunk


def run_dependency_build_parallel():
    """
    The main orchestrator function. Sets up and runs the parallel processing pipeline.
    """
    logger = logging.getLogger('DependencyBuilder')

    if RUN_IN_TEST_MODE:
        logger.warning("=" * 60)
        logger.warning(f"=== SCRIPT IS RUNNING IN TEST MODE ===")
        logger.warning(f"=== Processing only {NUM_TEST_SENTENCES} sentences. ===")
        logger.warning("=" * 60)

    logger.info("====== Starting Dependency Relation Set Construction (Parallel) ======")

    if DEPENDENCY_FILE.exists():
        logger.warning(f"Output file already exists: {DEPENDENCY_FILE}. Aborting to prevent overwrite.")
        return

    start_time = time.time()

    reader = WikiCorpusReader(WIKI_EXTRACTED_PATH, silent=True)

    # Get the sentences to process. If in test mode, only take a small slice.
    if RUN_IN_TEST_MODE:
        sentence_iterator_for_test = reader.stream_sentences()
        sentences_to_process = list(itertools.islice(sentence_iterator_for_test, NUM_TEST_SENTENCES))
        total_sentences = len(sentences_to_process)
        # We turn the list back into an iterator for the chunking function.
        sentence_iterator = iter(sentences_to_process)
        logger.info(f"Test mode: Processing {total_sentences:,} sentences.")
    else:
        # For a full run, do the pre-scan for an accurate progress bar.
        logger.info("Performing a quick pre-scan of the raw corpus to get an accurate sentence count...")
        reader_for_count = WikiCorpusReader(WIKI_EXTRACTED_PATH, silent=True)
        total_sentences = sum(1 for _ in reader_for_count.stream_sentences())
        logger.info(f"Full mode: Found {total_sentences:,} sentences to process.")
        sentence_iterator = reader.stream_sentences()

    final_relations_counter = Counter()

    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        logger.info(f"Starting worker pool with {NUM_PROCESSES} processes.")

        chunk_generator = read_sentences_in_chunks(sentence_iterator, CHUNK_SIZE)
        total_chunks = (total_sentences + CHUNK_SIZE - 1) // CHUNK_SIZE

        logger.info(f"Distributing {total_chunks:,} chunks to workers for parsing...")

        pbar = tqdm(pool.imap_unordered(process_sentence_chunk, chunk_generator), total=total_chunks,
                    desc="Parsing Chunks")

        for local_counter in pbar:
            final_relations_counter.update(local_counter)
            pbar.set_postfix({"total_unique_relations": f"{len(final_relations_counter):,}"})

    logger.info(f"All chunks parsed. Found {len(final_relations_counter):,} unique, filtered dependency relations.")

    # 1. PRUNE: Filter the counter to keep only relations that meet the minimum count.
    logger.info(f"Pruning relations with count < {MIN_RELATION_COUNT}...")
    pruned_relations = {
        rel for rel, count in final_relations_counter.items()
        if count >= MIN_RELATION_COUNT
    }
    logger.info(f"Pruned from {len(final_relations_counter):,} to {len(pruned_relations):,} unique relations.")

    # 2. BUILD INDEX: Convert the pruned set into an efficient inverted index.
    logger.info("Building inverted index from pruned relations...")
    inverted_index = defaultdict(list)
    # Iterate through the smaller, pruned set.
    for rel_tuple in tqdm(pruned_relations, desc="Indexing Relations"):
        dep, head, child = rel_tuple
        # Add the relation to the list for both the head and the child word.
        inverted_index[head].append(rel_tuple)
        inverted_index[child].append(rel_tuple)

    logger.info(f"Final index contains {len(inverted_index):,} unique word keys.")

    # 3. SAVE: Save the final inverted index to the file.
    logger.info(f"Saving final inverted index to {DEPENDENCY_FILE} using pickle...")
    try:
        with open(DEPENDENCY_FILE, 'wb') as f_out:
            # We convert defaultdict to a regular dict for saving to avoid potential issues
            # when loading in environments without defaultdict imported.
            pickle.dump(dict(inverted_index), f_out)
        logger.info(f"✅ Successfully saved dependency index data.")
    except Exception as e:
        logger.error(f"Failed to save data: {e}", exc_info=True)

    duration = time.time() - start_time
    if RUN_IN_TEST_MODE:
        logger.info(f"Test run finished in {duration:.2f} seconds.")
    else:
        logger.info(f"Full dependency set construction finished in {duration / 3600:.2f} hours.")

    logger.info("====== Dependency Relation Set Construction Finished ======")


if __name__ == '__main__':
    setup_logging()
    run_dependency_build_parallel()
