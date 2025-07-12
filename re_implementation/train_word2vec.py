import logging
from pathlib import Path
from datetime import datetime
from gensim.models import word2vec
from gensim.models.word2vec import LineSentence
import time
import MeCab
import matplotlib.pyplot as plt
import seaborn as sns

from wiki_corpus_reader.wiki_corpus_reader import WikiCorpusReader

WIKI_EXTRACTED_PATH = "/Volumes/T7/Bachelorthesis/jawiki_data/jawiki_extracted"
PROCESSED_DATA_DIR = Path("processed_corpus_data")
SIMPLE_TOKENIZED_CORPUS_FILE = PROCESSED_DATA_DIR / "simple_jawiki_tokenized.txt"
MODEL_DIR = Path("model")
FIGURES_DIR = Path("figures")
MODEL_FILENAME = "jawiki_min_count_5.word2vec.model"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME

def setup_logging():
    """
    Sets up a more advanced logging configuration.
    - Console handler shows all INFO messages, including progress and gensim.
    - File handler shows INFO from our own code but filters out gensim's INFO messages 
      and repetitive progress messages.
    """
    log_dir = Path("logs/word2vec_training")
    log_dir.mkdir(exist_ok=True)
    
    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get the root logger.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO) # Set the lowest level to capture all messages.
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # --- 1. Configure the Console Handler ---
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    # --- 2. Configure the File Handler ---
    file_handler = logging.FileHandler(
        log_dir / f"word2vec_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        encoding='utf-8'
    )
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)

    # This filter will reject log records that we don't want in the file
    class CleanLogFileFilter(logging.Filter):
        def filter(self, record):
            # Filter out gensim INFO messages
            is_gensim_info = record.name.startswith('gensim') and record.levelno == logging.INFO
            
            # Filter out progress messages from CorpusPreprocessor (like "Processed 100,000 sentences...")
            is_progress_message = (
                record.name == 'CorpusPreprocessor' and 
                record.levelno == logging.INFO and
                'sentences...' in record.getMessage()
            )
            
            return not (is_gensim_info or is_progress_message)

    file_handler.addFilter(CleanLogFileFilter())
    root_logger.addHandler(file_handler)

    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    logging.info("Logging configured. Console will show all progress. File will be clean (no progress/gensim spam).")

def preprocess_and_save_corpus():
    """
    Pre-tokenizes the entire corpus using an intelligent, look-ahead logic with MeCab
    and saves the result to a text file. This ensures that conjugated words are kept
    whole (e.g., '食べた' instead of '食べ た').
    
    Returns:
        Path: Path to the saved tokenized corpus file.
        float: Time taken for preprocessing in seconds.
    """
    logger = logging.getLogger('W2V_Preprocessor')
    
    if SIMPLE_TOKENIZED_CORPUS_FILE.exists():
        logger.info(f"Pre-tokenized corpus already exists at: {SIMPLE_TOKENIZED_CORPUS_FILE}")
        logger.info("Delete this file if you want to re-preprocess the corpus.")
        return SIMPLE_TOKENIZED_CORPUS_FILE, 0.0
    
    PROCESSED_DATA_DIR.mkdir(exist_ok=True)
    
    logger.info("Starting corpus preprocessing with intelligent tokenization...")
    logger.info(f"Output file will be: {SIMPLE_TOKENIZED_CORPUS_FILE}")
    
    if not Path(WIKI_EXTRACTED_PATH).exists():
        raise FileNotFoundError(f"Corpus path not found: {WIKI_EXTRACTED_PATH}")
    
    reader = WikiCorpusReader(WIKI_EXTRACTED_PATH)
    
    try:
        tagger = MeCab.Tagger('')
    except RuntimeError as e:
        logger.error(f"Failed to initialize MeCab Tagger. Error: {e}", exc_info=True)
        raise

    # Define constants and rules for parsing, mirroring CorpusProcessor
    MECAB_BOS_NODE, MECAB_EOS_NODE = 1, 2
    idx_pos1 = 0
    
    # Define rules for word reconstruction and filtering
    reconstruct_base_pos = {'動詞', '形容詞'} # 動詞 (Verb), 形容詞 (Adjective)
    attachable_pos = {'助動詞'} # 助動詞 (Auxiliary Verb)
    excluded_pos = [
        '補助記号',   # Supplementary Symbol (e.g., 、 。 （ ） - comma, period, parentheses)
        '助詞',       # Particle (e.g., は, が, を - grammatical markers like 'wa', 'ga', 'o')
        '接尾辞',     # Suffix (e.g., -的, -化, -者 - creates words like '-like', '-ization', '-person')
        '接頭辞',     # Prefix (e.g., 御-, 無-, 非- - honorifics or negation prefixes like 'go-', 'mu-')
        '感動詞',     # Interjection (e.g., ああ, えっ - "ah", "eh?!")
        'フィラー',   # Filler (e.g., あの, えっと - "um", "uhh")
        'その他',     # Other (a catch-all for unclassified items)
        '空白',       # Whitespace
        '記号',       # Symbol (general category for other symbols like mathematical operators)
        # Note: '助動詞' (Auxiliary Verb) is NOT on this list because it's handled by the
        # look-ahead logic.
    ]
    preprocessing_start_time = time.time()
    sentence_count = 0
    
    with open(SIMPLE_TOKENIZED_CORPUS_FILE, 'w', encoding='utf-8') as f:
        logger.info("Streaming sentences from corpus and tokenizing with MeCab...")
        
        for sentence_text in reader.stream_sentences():
            tagger.parse('') # Reset tagger
            node = tagger.parseToNode(sentence_text)
            
            reconstructed_tokens = []
            while node:
                # Skip BOS/EOS nodes and empty surfaces
                if not node.surface.strip() or node.stat in (MECAB_BOS_NODE, MECAB_EOS_NODE):
                    node = node.next
                    continue
                
                features = node.feature.split(',')
                if len(features) <= idx_pos1:
                    node = node.next
                    continue

                pos = features[idx_pos1]

                # If the token is a particle, symbol, suffix, etc., skip it entirely.
                if pos in excluded_pos:
                    node = node.next
                    continue
                
                # This is a potential base token (Noun, Verb, Adjective, etc.).
                # Start with its surface form.
                current_surface = node.surface
                
                # If it's a verb or adjective, look ahead for auxiliaries to combine.
                if pos in reconstruct_base_pos:
                    lookahead_node = node.next
                    while lookahead_node:
                        if not lookahead_node.surface.strip() or lookahead_node.stat in (MECAB_BOS_NODE, MECAB_EOS_NODE):
                            break
                        
                        next_features = lookahead_node.feature.split(',')
                        if len(next_features) > idx_pos1 and next_features[idx_pos1] in attachable_pos:
                            # It's an attachable part. Append its surface and consume the node.
                            current_surface += lookahead_node.surface
                            node = lookahead_node
                            lookahead_node = node.next
                        else:
                            break # Not an attachable part, stop looking.
                
                reconstructed_tokens.append(current_surface)
                
                # Advance to the next node in the original sequence.
                node = node.next

            # If we found any valid tokens in the sentence, write them to the file.
            if reconstructed_tokens:
                f.write(' '.join(reconstructed_tokens) + '\n')
                sentence_count += 1
                
                if sentence_count % 100000 == 0:
                    logger.info(f"Processed {sentence_count:,} sentences...")
    
    preprocessing_end_time = time.time()
    preprocessing_duration = preprocessing_end_time - preprocessing_start_time
    
    logger.info(f"Preprocessing complete!")
    logger.info(f"Total sentences written to file: {sentence_count:,}")
    logger.info(f"Time taken: {preprocessing_duration / 60:.2f} minutes")
    logger.info(f"Tokenized corpus saved to: {SIMPLE_TOKENIZED_CORPUS_FILE}")
    
    file_size_mb = SIMPLE_TOKENIZED_CORPUS_FILE.stat().st_size / (1024 * 1024)
    logger.info(f"File size: {file_size_mb:.1f} MB")
    
    return SIMPLE_TOKENIZED_CORPUS_FILE, preprocessing_duration

def create_timing_visualization(timings, output_path):
    """
    Creates and saves a bar chart visualizing the time spent in different stages.
    
    Args:
        timings (dict): A dictionary with stage names as keys and durations in seconds as values.
        output_path (Path): The path to save the generated image file.
    """
    labels = list(timings.keys())
    values = [t / 60 for t in timings.values()] # Convert seconds to minutes

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = sns.barplot(x=labels, y=values, ax=ax, palette="viridis", hue=labels, legend=False)
    
    ax.set_title('Word2Vec Training Pipeline - Time Analysis', fontsize=16, fontweight='bold')
    ax.set_ylabel('Time (minutes)', fontsize=12)
    ax.set_xlabel('Processing Stage', fontsize=12)

    # Add text labels on top of each bar
    for bar in bars.patches:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{bar.get_height():.2f} min',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )

    # Rotate x-axis labels if they're long
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    logging.info(f"📊 Timing visualization saved to: {output_path}")
    plt.close(fig)

def train_word2vec_model(tokenized_corpus_file):
    """
    Train the Word2Vec model using the pre-tokenized corpus file.
    
    Args:
        tokenized_corpus_file (Path): Path to the tokenized corpus file
        
    Returns:
        tuple: (trained_model, training_duration_in_seconds)
    """
    logger = logging.getLogger('Word2VecTrainer')
    
    # Define parameters
    vector_size = 400 # As per paper
    window_size = 5 # As per paper
    min_word_count = 5
    num_workers = 11 # CPU cores used for training
    epochs = 5 # Default gensim value
    
    logger.info("Initializing Word2Vec training...")
    logger.info(f"Parameters: sg=0 (CBOW), vector_size={vector_size}, window={window_size}, min_count={min_word_count}, workers={num_workers}, epochs={epochs}")
    
    # Use LineSentence for efficient reading from file
    # LineSentence expects one sentence per line with space-separated tokens
    sentences = LineSentence(str(tokenized_corpus_file))
    logger.info(f"Initialized LineSentence reader for: {tokenized_corpus_file}")
    
    logger.info("Starting Word2Vec model training...")
    training_start_time = time.time()
    
    model = word2vec.Word2Vec(
        sentences=sentences,
        sg=0,  # 0 for CBOW (as per paper)
        vector_size=vector_size,
        window=window_size,
        min_count=min_word_count,
        workers=num_workers,
        epochs=epochs
    )
    
    training_end_time = time.time()
    training_duration = training_end_time - training_start_time
    
    logger.info(f"Word2Vec training completed in {training_duration / 60:.2f} minutes")
    
    return model, training_duration

def train_and_save_model():
    """
    Orchestrates the entire word2vec model training pipeline with optimized preprocessing.
    """
    logger = logging.getLogger('Word2VecPipeline')
    logger.info("====== Starting Optimized Word2Vec Model Training Pipeline ======")

    MODEL_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    # --- Step 1: Preprocess and tokenize corpus (only done once) ---
    logger.info("--- Step 1: Corpus Preprocessing ---")
    tokenized_file, preprocessing_duration = preprocess_and_save_corpus()
    
    # --- Step 2: Train Word2Vec model ---
    logger.info("--- Step 2: Word2Vec Training ---")
    model, training_duration = train_word2vec_model(tokenized_file)
    
    # --- Step 3: Save the trained model ---
    logger.info("--- Step 3: Saving Model ---")
    save_start_time = time.time()
    logger.info(f"Saving trained model to: {MODEL_PATH}")
    model.save(str(MODEL_PATH))
    save_duration = time.time() - save_start_time
    logger.info(f"Model saved successfully in {save_duration:.2f} seconds")
    
    # --- Step 4: Performance Summary ---
    total_duration = preprocessing_duration + training_duration + save_duration
    
    logger.info("\n" + "="*50)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("="*50)
    logger.info(f"Corpus Preprocessing: {preprocessing_duration / 60:.2f} minutes")
    logger.info(f"Model Training: {training_duration / 60:.2f} minutes") 
    logger.info(f"Model Saving: {save_duration:.2f} seconds")
    logger.info(f"Total Pipeline Time: {total_duration / 60:.2f} minutes")
    logger.info("="*50)
    
    # --- Step 5: Create timing visualization ---
    timings = {
        "Corpus Preprocessing": preprocessing_duration,
        "Word2Vec Training": training_duration,
        "Model Saving": save_duration
    }
    viz_path = FIGURES_DIR / f"word2vec_pipeline_timing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    create_timing_visualization(timings, viz_path)

    # --- Step 6: Test the trained model ---
    logger.info("\n--- Step 6: Testing the trained model ---")
    try:
        vocab_size = len(model.wv.key_to_index)
        logger.info(f"Model vocabulary size: {vocab_size:,} words")
        
        # Test with some Japanese words
        test_words = ["猫", "東京", "走る", "美しい", "日本", "学校"]
        for word in test_words:
            if word in model.wv:
                similar_words = model.wv.most_similar(word, topn=3)
                logger.info(f"Words most similar to '{word}': {similar_words}")
            else:
                logger.warning(f"Test word '{word}' not in model vocabulary (likely below min_count={min_word_count})")
                
    except Exception as e:
        logger.error(f"An error occurred during model testing: {e}")

    logger.info("\n====== Word2Vec Model Training Pipeline Finished Successfully ======")
    
    return model

if __name__ == '__main__':
    setup_logging()
    trained_model = train_and_save_model()