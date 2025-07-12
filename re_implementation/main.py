import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import copy
import sys
import MeCab

# --- Component Imports ---
from wiki_corpus_reader.wiki_corpus_reader import WikiCorpusReader
from corpus_processor.corpus_processor import CorpusProcessor
from distractor_generators.baseline_generator.baseline_generator import BaselineGenerator
from distractor_generators.spelling_generator.spelling_generator import SpellingGenerator
from distractor_generators.similarity_generator.similarity_generator import SimilarityGenerator
from distractor_generators.cooccurrence_generator.cooccurrence_generator import CooccurrenceGenerator
from distractor_filtering.distractor_filter import DistractorFilter

# --- Build Script Imports ---
from train_word2vec import train_and_save_model
from build_cooccurrence_data import run_cooccurrence_calculation
from build_trigram_data import run_trigram_build_parallel as run_trigram_build
from build_dependency_data import run_dependency_build_parallel as run_dependency_build

# --- Configuration ---
WIKI_EXTRACTED_PATH = "/Volumes/T7/Bachelorthesis/jawiki_data/jawiki_extracted"
MODEL_DIR = Path("model")
DATA_DIR = Path("processed_corpus_data")
FIGURES_DIR = Path("figures")

# --- File Paths ---
W2V_MODEL_PATH = MODEL_DIR / "jawiki_min_count_5.word2vec.model"
SIMPLE_TOKENIZED_CORPUS_FILE = DATA_DIR / "simple_jawiki_tokenized.txt"
WORD_DETAILS_FILE = DATA_DIR / "jp_all_words_details.json"
COOCCURRENCE_FILE = DATA_DIR / "jp_cooccurrence_counts.json"
TRIGRAM_FILE = DATA_DIR / "jp_trigram_counts.pkl"
DEPENDENCY_FILE = DATA_DIR / "jp_dependency_relations.pkl"

TOTAL_SENTENCES = 25_408_585
BLANK_PLACEHOLDER = "___"

def setup_logging():
    """
    Sets up a robust logging configuration for the entire application,
    outputting to both the console and a time-stamped log file.
    """
    log_dir = Path("logs") / "distractor_generation"
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
        log_dir / f"distractor_generation_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", 
        encoding='utf-8'
    )
    file_handler.setFormatter(log_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    logging.info("Logging configured to output to console and file.")

def run_all_preprocessing_if_needed():
    """
    Checks for all necessary pre-processed files and runs the corresponding
    build scripts if any are missing.
    """
    main_logger = logging.getLogger('Main.Setup')
    main_logger.info("--- Checking for all required pre-processed data files ---")

    files_to_check = {
        "Word2Vec Model": W2V_MODEL_PATH,
        "Tokenized Corpus": SIMPLE_TOKENIZED_CORPUS_FILE,
        "Word Details JSON": WORD_DETAILS_FILE,
        "Co-occurrence Counts": COOCCURRENCE_FILE,
        "Trigram Set": TRIGRAM_FILE,
        "Dependency Index": DEPENDENCY_FILE,
    }

    missing_files = [name for name, path in files_to_check.items() if not path.exists()]

    if not missing_files:
        main_logger.info("✅ All pre-processed data files found. Skipping build steps.")
        return True

    main_logger.warning(f"Missing pre-processed files: {', '.join(missing_files)}. Starting build process...")

    try:
        if not W2V_MODEL_PATH.exists() or not SIMPLE_TOKENIZED_CORPUS_FILE.exists():
            main_logger.info("\n--- Running Word2Vec Training Pipeline ---")
            train_and_save_model()

        # This assumes corpus_processor.py has been run manually on the full corpus to create this file. Otherwise we would run it twice (once for the word2vec and once for the following data)
        if not WORD_DETAILS_FILE.exists():
             main_logger.error(f"{WORD_DETAILS_FILE} is missing. The system cannot run without it. "
                             "Please run corpus_processor.py manually on the full corpus first.")
             raise FileNotFoundError(f"Missing critical file for generators: {WORD_DETAILS_FILE}")

        if not COOCCURRENCE_FILE.exists():
            main_logger.info("\n--- Running Co-occurrence Data Build ---")
            run_cooccurrence_calculation()

        if not TRIGRAM_FILE.exists():
            main_logger.info("\n--- Running Trigram Data Build ---")
            run_trigram_build()

        if not DEPENDENCY_FILE.exists():
            main_logger.info("\n--- Running Dependency Data Build ---")
            run_dependency_build()

    except Exception as e:
        main_logger.critical(f"A preprocessing step failed: {e}", exc_info=True)
        return False

    main_logger.info("\n--- ✅ All required preprocessing steps completed successfully. ---")
    return True

MECAB_EXCLUDED_POS = {
    '補助記号', '助詞', '接尾辞', '感動詞',
    'フィラー', 'その他', '空白', '記号', '接頭辞',
}
# Note: '助動詞' (Auxiliary Verb) is NOT in the exclusion list,
# because it needs to be seen by the look-ahead logic.

def get_trigram_context(sentence_with_blank, tagger, placeholder=BLANK_PLACEHOLDER):
    """
    Tokenizes text around a blank, applying look-ahead logic for verbs/adjectives
    and filtering non-content words to match the build process.
    """
    try:
        pre_blank, post_blank = sentence_with_blank.split(placeholder, 1)
        
        def tokenize_with_lookahead(text):
            reconstructed_tokens = []
            node = tagger.parseToNode(text)
            while node:
                if not node.surface.strip() or node.stat in [1, 2]:
                    node = node.next
                    continue
                
                features = node.feature.split(',')
                pos = features[0]

                if pos in MECAB_EXCLUDED_POS:
                    node = node.next
                    continue
                
                current_surface = node.surface
                
                # The Look-ahead Logic
                if pos in {'動詞', '形容詞'}:
                    lookahead_node = node.next
                    while lookahead_node:
                        if not lookahead_node.surface.strip() or lookahead_node.stat in [1, 2]:
                            break
                        
                        next_features = lookahead_node.feature.split(',')
                        if next_features and next_features[0] == '助動詞':
                            current_surface += lookahead_node.surface
                            node = lookahead_node
                            lookahead_node = node.next
                        else:
                            break
                
                reconstructed_tokens.append(current_surface)
                node = node.next
            return reconstructed_tokens

        pre_tokens = tokenize_with_lookahead(pre_blank)
        post_tokens = tokenize_with_lookahead(post_blank)

        prev_word = pre_tokens[-1] if pre_tokens else "BOS"
        next_word = post_tokens[0] if post_tokens else "EOS"
        
        return prev_word, next_word
    except Exception as e:
        logging.getLogger('Main').error(f"Could not extract trigram context: {e}", exc_info=True)
        return None, None

def create_corpus_comparison_plot(reader_stats, output_dir="figures"):
    """
    Generates a standalone plot comparing the corpus size with the reference paper.
    
    Args:
        reader_stats (dict): Statistics dictionary from WikiCorpusReader.
        output_dir (str): Directory to save the plot.
    """
    logger = logging.getLogger('Main.Visualization')
    logger.info("Creating corpus size comparison plot...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Set plot style and create a figure
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle('Japanese Wikipedia Corpus Analysis Summary', fontsize=18, fontweight='bold')
    
    # --- Data for the plot ---
    paper_sentences = 14_000_000
    our_sentences = reader_stats.get('total_sentences_found', 0)
    
    labels = ['Paper (Chinese)', 'Our Corpus (Japanese)']
    values = [paper_sentences, our_sentences]
    colors = ['lightcoral', 'mediumseagreen']
    
    # Create the bar plot using Seaborn
    bars = sns.barplot(x=labels, y=values, ax=ax, palette=colors, hue=labels, legend=False)
    
    # --- Formatting the plot to match the image ---
    ax.set_title('Corpus Size Comparison', fontsize=16)
    ax.set_ylabel('Number of Sentences', fontsize=14)
    ax.set_xlabel('') # No x-axis label needed
    
    # Format y-axis ticks to millions (e.g., 15.0M)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    # Set y-axis limits to match the desired scale
    ax.set_ylim(0, 27_000_000)
    
    # Add text labels on top of each bar
    for bar in bars.patches:
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # x-position (center of the bar)
            bar.get_height(),                  # y-position (top of the bar)
            f'{bar.get_height()/1e6:.1f}M',     # The text to display, formatted to millions
            ha='center',                       # Horizontal alignment
            va='bottom',                       # Vertical alignment
            fontsize=14,
            fontweight='bold',
            color='black'
        )
        
    # Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    plot_file_path = output_path / "corpus_size_comparison.png"
    plt.savefig(plot_file_path, dpi=300)
    logger.info(f"📊 Corpus size comparison plot saved to: {plot_file_path.resolve()}")
    plt.close(fig)

if __name__ == "__main__":
    setup_logging()
    main_logger = logging.getLogger('Main')
    main_logger.info("====== Distractor Generation System Starting ======")

    # --- Step 1: Run all preprocessing if needed ---
    if not run_all_preprocessing_if_needed():
        main_logger.critical("Preprocessing failed or essential files are missing. Exiting.")
        sys.exit(1)

    # --- Step 2: Load Data and Initialize Generators & Filters ---
    main_logger.info("\n--- Loading data and initializing generators & filters ---")
    corpus_processor_instance = CorpusProcessor()

    all_words_data = CorpusProcessor.load_all_words_details(str(WORD_DETAILS_FILE))
    if not all_words_data:
        main_logger.critical("Failed to load processed corpus data. Cannot continue.")
        sys.exit(1)

    # Initialize Generators
    generators = {
        "Baseline": BaselineGenerator(corpus_processor_instance, all_words_data),
        "Spelling": SpellingGenerator(corpus_processor_instance, all_words_data),
        "Similarity": SimilarityGenerator(W2V_MODEL_PATH, corpus_processor_instance),
    }
    
    # Load Co-occurrence generator separately as it's large and might be optional.
    cooccurrence_data = CooccurrenceGenerator.load_cooccurrence_data(str(COOCCURRENCE_FILE))
    if cooccurrence_data:
        generators["Co-occurrence"] = CooccurrenceGenerator(
            corpus_processor_instance, all_words_data, cooccurrence_data, TOTAL_SENTENCES
        )

    # Initialize DistractorFilter
    main_logger.info("Initializing DistractorFilter...")
    distractor_filter = DistractorFilter(
        trigram_path=TRIGRAM_FILE,
        dependency_index_path=DEPENDENCY_FILE
    )
    main_logger.info("DistractorFilter is ready.")

    # --- Step 3: Demonstrate Comparative Distractor Generation with Filtering ---
    test_items = [
    # =================================================================================
    # CATEGORY 1: The Impact of an OPEN vs. CLOSED Carrier Sentence
    # =================================================================================
    
    # --- Pair 1: 猫 (cat) ---
    {
        "type": "Open Context",
        "target": "猫", 
        "sentence": f"私の{BLANK_PLACEHOLDER}はとても可愛い。" # Translation: My ___ is very cute.
    },
    {
        "type": "Closed Context",
        "target": "猫", 
        "sentence": f"その___は魚を咥えて、ニャーと鳴いた。" # Translation: That ___ was holding a fish and meowed "nyaa".
    },

    # --- Pair 2: 本 (book) ---
    {
        "type": "Open Context",
        "target": "本",
        "sentence": f"図書館で面白い{BLANK_PLACEHOLDER}を借りた。" # Translation: I borrowed an interesting ___ from the library.
    },
    {
        "type": "Closed Context",
        "target": "本",
        "sentence": f"彼はその___のページをめくり、活字を追い始めた。" # Translation: He turned the ___'s pages and began to follow the printed text.
    },

    # --- Pair 3: 走る (to run) ---
    {
        "type": "Open Context",
        "target": "走る",
        "sentence": f"彼は公園を{BLANK_PLACEHOLDER}のが好きだ。" # Translation: He likes to ___ in the park.
    },
    {
        "type": "Closed Context",
        "target": "走る",
        "sentence": f"彼は100メートルを11秒で{BLANK_PLACEHOLDER}ことができる。" # Translation: He can ___ 100 meters in 11 seconds.
    },

    # =================================================================================
    # CATEGORY 2: Testing Specific Generator Strengths with Good Sentences
    # =================================================================================
    
    {
        "type": "Testing Spelling/Similarity",
        "target": "政府", 
        "sentence": f"内閣総理大臣が率いる日本{BLANK_PLACEHOLDER}は、新しい法案を国会に提出した。" # Translation: The Japanese ___, led by the Prime Minister, submitted a new bill to the Diet.
    },
    {
        "type": "Testing Similarity (Antonyms)",
        "target": "寒い",
        "sentence": f"今日の朝は本当に{BLANK_PLACEHOLDER}ですね。" # Translation: It's really ___ this morning, isn't it? (This one is naturally constrained)
    },
    {
        "type": "Testing Co-occurrence/Thematic",
        "target": "コーヒー",
        "sentence": f"彼は朝食に___を一杯飲みます。" # Translation: He drinks a cup of ___ for breakfast. (The word 'cup' provides a good constraint)
    },

    # =================================================================================
    # CATEGORY 3: Testing Conjugation and Lemmatization
    # =================================================================================
    {
        "type": "Conjugation (Past Tense)",
        "target": "書いた",
        "sentence": f"彼女は昨日、長い手紙を{BLANK_PLACEHOLDER}。" # Translation: She ___ a long letter yesterday.
    },
    {
        "type": "Conjugation (Past Tense)",
        "target": "走った",
        "sentence": f"彼はマラソン大会で速く{BLANK_PLACEHOLDER}。" # Translation: He ___ fast in the marathon.
    },

    # =================================================================================
    # CATEGORY 4: Testing Ambiguous Single-Kanji Nouns with Good Sentences
    # =================================================================================
    {
        "type": "Ambiguous Kanji",
        "target": "日", 
        "sentence": f"次の日曜は父の{BLANK_PLACEHOLDER}なので、お墓参りに行きます。" # Translation: Next Sunday is Father's ___, so we will visit the grave. (Constraint: 'Father's ___' + 'grave visit' implies '命日' - anniversary of death, making '日' a good target)
    },
    {
        "type": "Ambiguous Kanji",
        "target": "金",
        "sentence": f"彼は銀行強盗で多額の現{BLANK_PLACEHOLDER}を手に入れた。" # Translation: He obtained a large amount of cash ___ in the bank robbery.
    },
    ]
    
    main_logger.info("="*60 + "\n")
    main_logger.info("DEMONSTRATING DISTRACTOR GENERATION WITH CHAINED FILTERING")
    main_logger.info("="*60 + "\n")

    # It is important to use the same tokenizer as we used in building the trigrams
    tagger_for_context = MeCab.Tagger('')

    # Test all Generators
    for item in test_items:
        word, sentence, type = item["target"], item["sentence"], item["type"]
        main_logger.info(f"▶️  ({type}) Target: '{word}' in Carrier Sentence: '{sentence}'")
        
        # Get the trigram context once per sentence.
        prev_word, next_word = get_trigram_context(sentence, tagger_for_context)
        if prev_word:
            main_logger.info(f"  (Context for Trigram Filter: ('{prev_word}', BLANK, '{next_word}'))")

        # Loop through all available generators to test each one.
        for name, generator in generators.items():
            if generator is None: 
                main_logger.info(f"  - {name:15s} (Filtered): [Generator not available]")
                continue

            # Generate more candidates than needed (e.g., 20) to give the filters options.
            raw_candidates = generator.generate_distractors(word, sentence, num_distractors=20)

            raw_candidates_set = set(raw_candidates)
            trigram_rejected_set = set()

            if prev_word:
                for candidate in raw_candidates:
                    if (prev_word, candidate, next_word) in distractor_filter.trigrams:
                        trigram_rejected_set.add(candidate)
            
            # 2. Get the set of candidates rejected by the Dependency filter.
            # We find the *accepted* ones first, then use set difference to find the rejected.
            dependency_accepted_list = distractor_filter.filter_by_dependency(raw_candidates, sentence)
            dependency_rejected_set = raw_candidates_set - set(dependency_accepted_list)

            # 3. Find the candidates that are in BOTH rejection sets.
            rejected_by_both_set = trigram_rejected_set.intersection(dependency_rejected_set)

            # 4. The final candidates are the raw candidates minus those rejected by both.
            final_candidates_list = [
                cand for cand in raw_candidates if cand not in rejected_by_both_set
            ]

            # Log the final, filtered results.
            main_logger.info(f"  - {name:15s} (Filtered - AND): {final_candidates_list[:5]}")
            main_logger.debug(f"    Raw: {raw_candidates[:10]}")
            main_logger.debug(f"    Trigram Rejects: {list(trigram_rejected_set)[:10]}")
            main_logger.debug(f"    Dependency Rejects: {list(dependency_rejected_set)[:10]}")
            main_logger.debug(f"    Final Rejects (In Both): {list(rejected_by_both_set)}")

        main_logger.info("-" * 60)


    main_logger.info("\n====== System finished demonstration. ======")
    sys.exit(0)