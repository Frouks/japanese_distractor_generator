import logging
from datetime import datetime
from pathlib import Path
from context_analyzer.context_analyzer import ContextAnalyzer
from corpus_processor.corpus_processor import CorpusProcessor
from distractor_generators.similarity_generator import SimilarityGenerator
from distractor_generators.baseline_generator import BaselineGenerator
from distractor_generators.spelling_generator import SpellingGenerator
from distractor_generators.cooccurrence_generator import CooccurrenceGenerator

MODEL_DIR = Path("model")
DATA_DIR = Path("processed_corpus_data")

TOTAL_SENTENCES = 25_408_585
BLANK_PLACEHOLDER = "___"
MECAB_EXCLUDED_POS = {
    '補助記号', '助詞', '接尾辞', '感動詞',
    'フィラー', 'その他', '空白', '記号', '接頭辞',
    # Note: '助動詞' (Auxiliary Verb) is NOT in the exclusion list,
    # because it needs to be seen by the look-ahead logic.
}

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


if __name__ == "__main__":
    setup_logging()