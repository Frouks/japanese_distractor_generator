import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import MeCab
import pandas as pd
from tqdm import tqdm

# This block allows the script to be run directly from the terminal
try:
    # Get the project root directory (the parent of the script's directory)
    project_root = Path(__file__).resolve().parent.parent
    # Add the project root to Python's path if it's not already there
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except NameError:
    # Fallback for environments where __file__ is not defined
    project_root = Path('.').resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from context_aware_distractor_generation_system.context_analyzer.context_analyzer import ContextAnalyzer
from context_aware_distractor_generation_system.corpus_processor.corpus_processor import CorpusProcessor
from context_aware_distractor_generation_system.distractor_filtering.distractor_filter import \
    DistractorFilter as CADGS_DistractorFilter
# CADGS Components
from context_aware_distractor_generation_system.distractor_generators.baseline_generator import \
    BaselineGenerator as CADGS_BaselineGenerator
from context_aware_distractor_generation_system.distractor_generators.contextual_generator import \
    ContextualGenerator as CADGS_ContextualGenerator
from context_aware_distractor_generation_system.distractor_generators.cooccurrence_generator import \
    CooccurrenceGenerator as CADGS_CooccurrenceGenerator
from context_aware_distractor_generation_system.distractor_generators.similarity_generator import \
    SimilarityGenerator as CADGS_SimilarityGenerator
from context_aware_distractor_generation_system.distractor_generators.spelling_generator import \
    SpellingGenerator as CADGS_SpellingGenerator
from re_implementation.distractor_filtering.distractor_filter import DistractorFilter as RI_DistractorFilter
# Re-Implementation Components
from re_implementation.distractor_generators.baseline_generator.baseline_generator import \
    BaselineGenerator as RI_BaselineGenerator
from re_implementation.distractor_generators.cooccurrence_generator.cooccurrence_generator import \
    CooccurrenceGenerator as RI_CooccurrenceGenerator
from re_implementation.distractor_generators.similarity_generator.similarity_generator import \
    SimilarityGenerator as RI_SimilarityGenerator
from re_implementation.distractor_generators.spelling_generator.spelling_generator import \
    SpellingGenerator as RI_SpellingGenerator

DATA_DIR = Path("../re_implementation/processed_corpus_data")

W2V_MODEL_PATH = Path("../re_implementation/model/jawiki_min_count_5.word2vec.model")
FAST_TEXT_MODEL = Path("../context_aware_distractor_generation_system/model/cc.ja.300.bin")
BERT_MODEL = "cl-tohoku/bert-base-japanese-whole-word-masking"

WORD_DETAILS_FILE = DATA_DIR / "jp_all_words_details.json"
COOCCURRENCE_FILE = DATA_DIR / "jp_cooccurrence_counts.json"
TRIGRAM_FILE = DATA_DIR / "jp_trigram_counts.pkl"
DEPENDENCY_FILE = DATA_DIR / "jp_dependency_relations.pkl"
TEST_CASES_FILE = Path("test_cases.csv")
OUTPUT_FILE = Path(f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

TOTAL_SENTENCES = 25_408_585
ENTROPY_THRESHOLD = 4.5
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
    log_dir.mkdir(parents=True, exist_ok=True)

    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    logging.getLogger('gensim.models._fasttext_bin').setLevel(logging.CRITICAL)

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


def initialize_re_implementation_system(corpus_processor: CorpusProcessor, all_words_data):
    """Loads all components for the Re-Implementation system."""
    logging.info("--- Initializing Re-Implementation System ---")

    if not all_words_data:
        raise FileNotFoundError(f"Could not load word details file: {WORD_DETAILS_FILE}")

    cooccurrence_data = RI_CooccurrenceGenerator.load_cooccurrence_data(str(COOCCURRENCE_FILE))

    generators = {
        "Baseline": RI_BaselineGenerator(corpus_processor, all_words_data),
        "Spelling": RI_SpellingGenerator(corpus_processor, all_words_data),
        "Similarity": RI_SimilarityGenerator(str(W2V_MODEL_PATH), corpus_processor),
        "Co-occurrence": RI_CooccurrenceGenerator(corpus_processor, all_words_data, cooccurrence_data,
                                                  TOTAL_SENTENCES) if cooccurrence_data else None,
    }

    distractor_filter = RI_DistractorFilter(trigram_path=TRIGRAM_FILE, dependency_index_path=DEPENDENCY_FILE)

    logging.info("--- Re-Implementation System Ready ---\n")
    return {"generators": generators, "filter": distractor_filter}


def initialize_cadgs_system(corpus_processor: CorpusProcessor, all_words_data):
    """Loads all components for the Context-Aware Distractor Generation System."""
    logging.info("--- Initializing Context-Aware Distractor Generation System (CADGS) ---")

    if not all_words_data:
        raise FileNotFoundError(f"Could not load word details file: {WORD_DETAILS_FILE}")

    cooccurrence_data = CADGS_CooccurrenceGenerator.load_cooccurrence_data(str(COOCCURRENCE_FILE))

    generators = {
        "Baseline": CADGS_BaselineGenerator(corpus_processor, all_words_data),
        "Spelling": CADGS_SpellingGenerator(corpus_processor, all_words_data),
        "Similarity": CADGS_SimilarityGenerator(str(FAST_TEXT_MODEL)),
        "Co-occurrence": CADGS_CooccurrenceGenerator(corpus_processor, all_words_data, cooccurrence_data,
                                                     TOTAL_SENTENCES) if cooccurrence_data else None,
        "Contextual": CADGS_ContextualGenerator(BERT_MODEL),
    }

    distractor_filter = CADGS_DistractorFilter(
        trigram_path=TRIGRAM_FILE,
        dependency_index_path=DEPENDENCY_FILE,
        bert_model_name=BERT_MODEL
    )

    context_analyzer = ContextAnalyzer(BERT_MODEL, ENTROPY_THRESHOLD)

    logging.info("--- CADGS Ready ---\n")
    return {"generators": generators, "filter": distractor_filter, "analyzer": context_analyzer}


if __name__ == "__main__":
    # We need to run the script as a module -> python -m evaluation_pipeline.evaluation_pipeline
    setup_logging()
    main_logger = logging.getLogger('EvaluationPipeline')
    main_logger.info("====== Evaluation pipeline started ======")
    logging.info("--- Initializing Corpus Processor ---")
    corpus_processor_instance = CorpusProcessor()

    logging.info("--- Loading all words data ---")
    all_words_data = CorpusProcessor.load_all_words_details(str(WORD_DETAILS_FILE))
    # --- Load Systems ---
    ri_system = initialize_re_implementation_system(corpus_processor=corpus_processor_instance,
                                                    all_words_data=all_words_data)
    cadgs_system = initialize_cadgs_system(corpus_processor=corpus_processor_instance, all_words_data=all_words_data)

    # --- Load Test Data ---
    if not TEST_CASES_FILE.exists():
        logging.error(f"Test cases file not found at: {TEST_CASES_FILE}")
        sys.exit(0)

    test_cases = pd.read_csv(TEST_CASES_FILE)
    logging.info(f"Loaded {len(test_cases)} test cases from {TEST_CASES_FILE}")

    # --- MeCab Tagger for trigram context ---
    # This ensures consistency with the build process
    tagger = MeCab.Tagger('')

    evaluation_results = []

    # --- Main Evaluation Loop ---
    for index, row in tqdm(test_cases.iterrows(), total=len(test_cases), desc="Processing Test Cases"):
        logging.info(f"=== Distractor Generation for Test Case #{index} === \n")
        sentence = str(row['sentence'])
        target = str(row['target'])

        case_result = {
            "test_case_id": index,
            "sentence": sentence,
            "sentence_in_english": str(row['english_sentence']),
            "predicted_context": "",
            "target": target,
            "re_implementation_results": {},
            "cadgs_results": {}
        }

        # --- 1. Analyze Context (CADGS) ---
        masked_sentence_for_analyzer = sentence.replace(BLANK_PLACEHOLDER,
                                                        cadgs_system["analyzer"].tokenizer.mask_token)
        predicted_context = cadgs_system["analyzer"].analyze_context_by_entropy(masked_sentence_for_analyzer)
        case_result["predicted_context"] = predicted_context.name

        # --- 2. Run Re-Implementation System ---
        prev_word, next_word = get_trigram_context(sentence, tagger, BLANK_PLACEHOLDER)
        logging.info("=== Re_Implementation Generation === \n")
        for name, generator in ri_system["generators"].items():
            if generator is None: continue

            distractor_candidates = generator.generate_distractors(target, sentence, num_distractors=30)

            # Apply RI filter logic ("reject if both fail")
            _, trigram_rejected = ri_system["filter"].filter_by_trigram(distractor_candidates, prev_word, next_word)
            _, dep_rejected = ri_system["filter"].filter_by_dependency(distractor_candidates, sentence)

            rejected_by_any = set(trigram_rejected) | set(dep_rejected)
            final_distractors = [cand for cand in distractor_candidates if cand not in rejected_by_any]

            case_result["re_implementation_results"][name] = {
                "distractor_candidates": distractor_candidates,
                "rejected_by": {
                    "trigram": trigram_rejected,
                    "dependency": dep_rejected,
                },
                "final_distractors": final_distractors
            }

        logging.info("\n=== CADGS Generation ===")
        # --- 3. Run CADGS ---
        for name, generator in cadgs_system["generators"].items():
            if generator is None: continue

            if name == "Contextual":
                masked_sentence_for_generator = sentence.replace(BLANK_PLACEHOLDER, cadgs_system["generators"][
                    "Contextual"].tokenizer.mask_token)
                distractor_candidates = generator.generate_distractors(masked_sentence=masked_sentence_for_generator,
                                                                       target_word=target,
                                                                       context_type=predicted_context,
                                                                       top_n=30, include_prob_score=False)
            elif name == "Co-occurrence":
                distractor_candidates = generator.generate_distractors(target_word_surface=target,
                                                                       sentence_with_blank=sentence,
                                                                       context_type=predicted_context,
                                                                       num_distractors=30,
                                                                       include_pmi_score=False)
            elif name == "Similarity":
                distractor_candidates = generator.generate_distractors(target_word=target,
                                                                       context_type=predicted_context,
                                                                       top_n=30, num_candidates=200,
                                                                       include_sim_score=False)
            else:  # Baseline- and SpellingGenerator
                distractor_candidates = generator.generate_distractors(target_word_surface=target,
                                                                       sentence_with_blank=sentence, num_distractors=30)

            # Apply CADGS filter logic ("reject if any fail")
            _, trigram_rejected = cadgs_system["filter"].filter_by_trigram(distractor_candidates, prev_word, next_word)
            _, dep_rejected = cadgs_system["filter"].filter_by_dependency(distractor_candidates, sentence)
            _, bert_rejected = cadgs_system["filter"].filter_by_bert_fixed(distractor_candidates, sentence,
                                                                           predicted_context,
                                                                           target)

            rejected_by_any = set(trigram_rejected) | set(dep_rejected) | set(bert_rejected)
            final_distractors = [cand for cand in distractor_candidates if cand not in rejected_by_any]

            case_result["cadgs_results"][name] = {
                "distractor_candidates": distractor_candidates,
                "rejected_by": {
                    "trigram": trigram_rejected,
                    "dependency": dep_rejected,
                    "bert_pll": bert_rejected
                },
                "final_distractors": final_distractors
            }

        logging.info("\n === Generation for test case finished === \n")

        evaluation_results.append(case_result)

    # --- 4. Save Results ---
    logging.info(f"Evaluation complete. Saving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=4)
    logging.info("✅ Results saved successfully.")
