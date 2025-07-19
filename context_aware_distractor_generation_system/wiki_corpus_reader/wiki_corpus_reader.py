import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm


class WikiCorpusReader:
    """
    Reads raw Wikipedia corpus data from wikiextractor output.

    Responsibilities:
    1. Find and parse the JSON files.
    2. Clean the text and split it into sentences.
    3. Act as an iterable, streaming sentences one by one to save memory.
    4. Collect statistics about the reading process (file counts, sentence lengths, etc.).
    5. Generate visualizations for its own statistics.
    """

    def __init__(self, wiki_extracted_path, log_level=logging.INFO, silent=False):
        """
        Initialize the WikiCorpusReader.

        Args:
            wiki_extracted_path (str): Path to the wiki corpus folder
            log_level: Logging level (default: INFO)
        """
        self.wiki_corpus_path = Path(wiki_extracted_path)
        self.logger = logging.getLogger('WikiCorpusReader')
        self.stats = {}  # Will be populated by stream_sentences

        if not silent:
            self._initialize_logging(log_level)

    def _initialize_logging(self, log_level):
        """
        Setting up logging for both console and file output.
        """
        if not self.logger.hasHandlers():
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            self.logger.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            # Add console handler
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            # Add file handler
            fh = logging.FileHandler(log_dir / f"wiki_corpus_reader_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                                     encoding='utf-8')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        self.logger.info(f"Initialized reader for path: {self.wiki_corpus_path}")

    def _find_all_json_files(self):
        """
        Find ALL valid JSON files in the extracted Wikipedia directory.
        Excludes macOS system files and other artifacts.
        """
        self.logger.info(f"Searching for JSON files in: {self.wiki_corpus_path.absolute()}")
        if not self.wiki_corpus_path.exists():
            raise FileNotFoundError(f"Path does not exist: {self.wiki_corpus_path.absolute()}")

        json_files = list(self.wiki_corpus_path.glob("**/*.json"))
        valid_json_files = [
            f for f in json_files
            if not f.name.startswith('.') and f.stat().st_size > 10
        ]
        self.logger.info(f"Found {len(valid_json_files)} valid JSON files to process.")
        return valid_json_files

    def _simple_japanese_sentence_split(self, text, long_sentence_log_list):
        """
        Splits and cleanes text into a list of valid sentences.
        """
        if not text or len(text.strip()) < 3:
            return []

        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'\n+', ' ', text)

        sentences = re.split(r'[。！？]', text)

        valid_sentences_data = []
        for sentence_str in sentences:
            sentence_str = sentence_str.strip()
            # If the sentence is too short or too long we ignore it
            if len(sentence_str) < 8 or len(sentence_str) > 300:
                continue

            total_chars = len(sentence_str)
            japanese_chars = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', sentence_str))

            if total_chars > 0 and (japanese_chars / total_chars) < 0.3:
                continue

            if len(sentence_str) > 200 and len(long_sentence_log_list) < 10:
                long_sentence_log_list.append(sentence_str)

            valid_sentences_data.append({'text': sentence_str, 'length': len(sentence_str)})

        return valid_sentences_data

    def stream_sentences(self):
        """
        A generator that finds, reads, and yields processed sentences one by one.
        This is memory-efficient. It also populates the self.stats dictionary.
        """
        self.logger.info("Starting to stream sentences from corpus...")
        start_time = time.time()

        self.stats = {
            'files_processed': 0, 'files_with_errors': 0, 'total_articles': 0,
            'total_sentences': 0, 'sentence_lengths': [], 'long_sentences_log': []
        }

        json_files = self._find_all_json_files()

        for file_path in tqdm(json_files, desc="Reading Corpus Files"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            article = json.loads(line)
                            text_content = article.get('text', '')
                            if text_content:
                                sentences_info = self._simple_japanese_sentence_split(text_content,
                                                                                      self.stats['long_sentences_log'])
                                if sentences_info:
                                    self.stats['total_articles'] += 1
                                    for sent_info in sentences_info:
                                        self.stats['total_sentences'] += 1
                                        self.stats['sentence_lengths'].append(sent_info['length'])
                                        # yield pauses the function, sends back one piece of data (a sentence string), and waits. When the next piece of data is requested, it resumes from where it left off.
                                        yield sent_info['text']
                        except json.JSONDecodeError:
                            continue  # Skip malformed lines
                self.stats['files_processed'] += 1
            except Exception:
                self.stats['files_with_errors'] += 1
                continue

        end_time = time.time()
        self.stats['total_processing_time_seconds'] = end_time - start_time
        self.logger.info(f"Finished streaming. Processed {self.stats['total_sentences']:,} sentences.")
        self.logger.info(f"Corpus reading took {self.stats['total_processing_time_seconds']:.2f} seconds.")

    def get_corpus_statistics(self):
        """Calculates and returns a summary dictionary of the reading process."""
        if not self.stats or self.stats['total_sentences'] == 0:
            self.logger.warning("No statistics available. Run stream_sentences() first.")
            return {}

        s = self.stats
        sentence_lengths = s.get('sentence_lengths', [])
        lengths_arr = np.array(s['sentence_lengths'])
        summary = {
            'total_files_processed': s['files_processed'],
            'files_with_read_errors': s['files_with_errors'],
            'total_articles_found': s['total_articles'],
            'total_sentences_found': s['total_sentences'],
            'total_processing_time_seconds': s['total_processing_time_seconds'],
            'sentence_lengths': sentence_lengths,
            'sentence_length_stats': {
                'mean': np.mean(lengths_arr), 'median': np.median(lengths_arr),
                'min': np.min(lengths_arr), 'max': np.max(lengths_arr),
                'std_dev': np.std(lengths_arr)
            }
        }
        return summary

    def generate_and_save_visualizations(self, output_dir="figures"):
        """Creates and saves visualizations based on the collected corpus stats."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        stats = self.get_corpus_statistics()
        if not stats:
            self.logger.error("Cannot generate visualizations without statistics.")
            return

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))

        sentence_lengths_data = self.stats.get('sentence_lengths', [])
        if not sentence_lengths_data:
            self.logger.warning("No sentence length data to plot.")
            return

        sns.histplot(sentence_lengths_data, bins=50, ax=ax, color='cornflowerblue', kde=True)

        sl_stats = stats['sentence_length_stats']
        ax.axvline(sl_stats['mean'], color='crimson', linestyle='--', label=f"Mean: {sl_stats['mean']:.1f}")
        ax.axvline(sl_stats['median'], color='forestgreen', linestyle='--', label=f"Median: {sl_stats['median']:.1f}")

        ax.set_title('Distribution of Sentence Lengths in Japanese Wikipedia Corpus', fontsize=15)
        ax.set_xlabel('Sentence Length (characters)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend()

        plot_file_path = output_path / "corpus_sentence_length_distribution.png"
        plt.tight_layout()
        plt.savefig(plot_file_path, dpi=300)
        self.logger.info(f"📊 Corpus stats visualization saved to: {plot_file_path.resolve()}")
        plt.close(fig)


# ==============================================================================
#  TESTING BLOCK: This code runs only when you execute `python wiki_corpus_reader.py`
# ==============================================================================
if __name__ == '__main__':
    # --- Step 1: Set up logging ---
    test_output_base_dir = Path("wiki_corpus_reader_test_output")
    test_output_base_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = test_output_base_dir / "wiki_corpus_reader_test.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
        # Set up handlers to log to both the console and the specified file.
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
        ]
    )
    test_logger = logging.getLogger("WikiCorpusReaderTest")
    test_logger.info("=== Starting WikiCorpusReader Self-Test Script (on Real Corpus Sample) ===")
    test_logger.info(f"Log file for this test run is being saved to: {log_file_path.resolve()}")

    # --- Step 2: Define the path to the corpus ---
    WIKI_EXTRACTED_PATH = "/Volumes/T7/Bachelorthesis/jawiki_data/jawiki_extracted"

    if not Path(WIKI_EXTRACTED_PATH).exists():
        test_logger.error(f"Real corpus path not found: {WIKI_EXTRACTED_PATH}")
        test_logger.error("Please update the WIKI_EXTRACTED_PATH variable in the test script.")
        exit()

    # --- Step 3: Initialize the WikiCorpusReader ---
    test_logger.info("\n--- Initializing WikiCorpusReader with REAL corpus path ---")
    try:
        reader = WikiCorpusReader(WIKI_EXTRACTED_PATH)
    except Exception as e:
        test_logger.critical(f"Failed to initialize WikiCorpusReader: {e}", exc_info=True)
        exit()

    # --- Step 4: Create a new instance to test, but limit the files it finds ---
    NUM_FILES_TO_TEST = 1000
    test_logger.info(f"\n--- Preparing to test with the first {NUM_FILES_TO_TEST} real corpus files ---")

    all_real_files = reader._find_all_json_files()
    if len(all_real_files) < NUM_FILES_TO_TEST:
        test_logger.warning(
            f"Corpus has fewer than {NUM_FILES_TO_TEST} files. Testing with all {len(all_real_files)} files.")
        files_to_process = all_real_files
    else:
        files_to_process = all_real_files[:NUM_FILES_TO_TEST]

    # Now, when `stream_sentences` calls `self._find_all_json_files`, it will run our lambda function instead.
    reader._find_all_json_files = lambda: files_to_process

    # --- Step 5: Test the `stream_sentences` generator and collect statistics ---
    test_logger.info("\n--- Testing sentence streaming and statistics gathering on real data sample ---")

    # Consume the generator to force it to run completely.
    sentence_count = 0
    for sentence in reader.stream_sentences():
        sentence_count += 1

    # --- Step 6: Verify the results ---
    test_logger.info("\n--- Verifying Test Results ---")

    # 1. Get the final statistics.
    stats = reader.get_corpus_statistics()
    test_logger.info(f"Final statistics from processing {stats.get('total_files_processed')} files:")
    test_logger.info(f"  - Files Processed: {stats.get('total_files_processed')}")
    test_logger.info(f"  - Articles Found: {stats.get('total_articles_found')}")
    test_logger.info(f"  - Sentences Found: {stats.get('total_sentences_found')}")
    test_logger.info(f"  - Number of Sentence Lengths Recorded: {len(stats.get('sentence_lengths', []))}")
    test_logger.info(f"  - Sentence Length Stats (Mean/Median/Min/Max): {stats.get('sentence_length_stats')}")

    # 2. Assert that the process ran and found something.
    assert stats.get(
        'total_files_processed') == NUM_FILES_TO_TEST, f"Expected to process {NUM_FILES_TO_TEST} files, but stats show {stats.get('total_files_processed')}."
    test_logger.info("✅ 'total_files_processed' stat matches expected number.")

    assert stats.get(
        'total_sentences_found') > 0, "Processing ran but found zero valid sentences, which is unlikely for 1000 real files."
    test_logger.info(f"✅ Found a plausible number of sentences: {stats.get('total_sentences_found'):,}.")

    assert stats.get('total_articles_found') > 0, "Processing ran but found zero articles with valid sentences."
    test_logger.info(f"✅ Found a plausible number of articles: {stats.get('total_articles_found'):,}.")

    assert len(stats.get('sentence_lengths')) == stats.get(
        'total_sentences_found'), "The number of recorded sentence lengths does not match the total sentence count."
    test_logger.info("✅ Sentence length tracking is consistent.")

    # --- Step 7: Test visualization generation ---
    test_logger.info("\n--- Testing visualization generation with real data sample ---")
    figures_dir = test_output_base_dir / "figures_from_test_data"
    figures_dir.mkdir(parents=True, exist_ok=True)

    reader.generate_and_save_visualizations(output_dir=str(figures_dir))
    assert (figures_dir / "corpus_sentence_length_distribution.png").exists(), "Visualization file was not created."
    test_logger.info(f"✅ Visualization created successfully in '{figures_dir.resolve()}'.")

    test_logger.info("\n=== WikiCorpusReader Self-Test Script Finished Successfully ===")
