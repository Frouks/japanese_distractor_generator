import logging
import sys
from datetime import datetime
from pathlib import Path

import MeCab

from constants.SentenceContextEnum import SentenceContextEnum
from context_analyzer.context_analyzer import ContextAnalyzer
from corpus_processor.corpus_processor import CorpusProcessor
from distractor_filtering.distractor_filter import DistractorFilter
from distractor_generators.baseline_generator import BaselineGenerator
from distractor_generators.contextual_generator import ContextualGenerator
from distractor_generators.cooccurrence_generator import \
    CooccurrenceGenerator
from distractor_generators.similarity_generator import SimilarityGenerator
from distractor_generators.spelling_generator import SpellingGenerator

DATA_DIR = Path("../re_implementation/processed_corpus_data")
MODEL_DIR = Path("model")
FAST_TEXT_MODEL = MODEL_DIR / "cc.ja.300.bin"
WORD_DETAILS_FILE = DATA_DIR / "jp_all_words_details.json"
COOCCURRENCE_FILE = DATA_DIR / "jp_cooccurrence_counts.json"
TRIGRAM_FILE = DATA_DIR / "jp_trigram_counts.pkl"
DEPENDENCY_FILE = DATA_DIR / "jp_dependency_relations.pkl"
BERT_MODEL = "cl-tohoku/bert-base-japanese-whole-word-masking"

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
    log_dir.mkdir(exist_ok=True)

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


if __name__ == "__main__":
    setup_logging()
    main_logger = logging.getLogger('Main')
    main_logger.info("====== Distractor Generation System Starting (CADGS) ======")

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
        "Similarity": SimilarityGenerator(str(FAST_TEXT_MODEL)),
        "Contextual": ContextualGenerator(BERT_MODEL)
    }

    cooccurrence_data = CooccurrenceGenerator.load_cooccurrence_data(str(COOCCURRENCE_FILE))
    if cooccurrence_data:
        generators["Co-occurrence"] = CooccurrenceGenerator(
            corpus_processor_instance, all_words_data, cooccurrence_data, TOTAL_SENTENCES
        )

    main_logger.info("Initializing DistractorFilter...")
    distractor_filter = DistractorFilter(
        trigram_path=TRIGRAM_FILE,
        dependency_index_path=DEPENDENCY_FILE,
        bert_model_name=BERT_MODEL
    )
    main_logger.info("DistractorFilter is ready.")
    main_logger.info("Initializing ContextAnalyzer... \n")
    context_analyzer = ContextAnalyzer(
        BERT_MODEL,
        ENTROPY_THRESHOLD,
    )
    main_logger.info("\n ContextAnalyzer is ready. \n")

    # --- Step 3: Demonstrate Comparative Distractor Generation with Filtering ---
    test_items = [
        {
            "sentence": "私の___はとても可愛い。",
            "target": "猫",
            "english_sentence": "My ___ is very cute.",
        },
        {
            "sentence": "その___は魚を咥えて、ニャーと鳴いた。",
            "target": "猫",
            "english_sentence": "That ___ was holding a fish and meowed 'nyaa'.",
        },
        {
            "sentence": "図書館で面白い___を借りた。",
            "target": "本",
            "english_sentence": "I borrowed an interesting ___ from the library.",
        },
        {
            "sentence": "彼はその___のページをめくり、活字を追い始めた。",
            "target": "本",
            "english_sentence": "He turned the ___'s pages and began to follow the printed text.",
        },
        {
            "sentence": "彼は公園を___のが好きだ。",
            "target": "走る",
            "english_sentence": "He likes to ___ in the park.",
        },
        {
            "sentence": "彼は100メートルを11秒で___ことができる。",
            "target": "走る",
            "english_sentence": "He can ___ 100 meters in 11 seconds.",
        },
        {
            "sentence": "内閣総理大臣が率いる日本___は、新しい法案を国会に提出した。",
            "target": "政府",
            "english_sentence": "The Japanese ___, led by the Prime Minister, submitted a new bill to the Diet.",
        },
        {
            "sentence": "今日の朝は本当に___ですね。",
            "target": "寒い",
            "english_sentence": "It's really ___ this morning, isn't it?",
        },
        {
            "sentence": "彼は朝食に___を一杯飲みます。",
            "target": "コーヒー",
            "english_sentence": "He drinks a cup of ___ for breakfast.",
        },
        {
            "sentence": "彼女は昨日、長い手紙を___。",
            "target": "書いた",
            "english_sentence": "She ___ a long letter yesterday.",
        },
        {
            "sentence": "彼はマラソン大会で速く___。",
            "target": "走った",
            "english_sentence": "He ___ fast in the marathon.",
        },
        {
            "sentence": "次の日曜は父の___なので、お墓参りに行きます。",
            "target": "日",
            "english_sentence": "Next Sunday is Father's ___, so we will visit the grave.",
        },
        {
            "sentence": "彼は銀行強盗で多額の現___を手に入れた。",
            "target": "金",
            "english_sentence": "He obtained a large amount of cash ___ in the bank robbery.",
        }
    ]

    main_logger.info("=" * 60 + "\n")
    main_logger.info("DEMONSTRATING DISTRACTOR GENERATION WITH CHAINED FILTERING")
    main_logger.info("=" * 60 + "\n")

    # It is important to use the same tokenizer as we used in building the trigrams
    tagger_for_context = MeCab.Tagger('')
    # Test all Generators
    for item in test_items:
        target_word, sentence = item["target"], item["sentence"]
        masked_sentence = sentence.replace("___", "[MASK]")
        context: SentenceContextEnum = context_analyzer.analyze_context_by_entropy(masked_sentence)
        main_logger.info(f"▶️ Target Word: '{target_word}' in Carrier Sentence: '{sentence}' with context: '{context}'")

        # Get the trigram context once per sentence.
        prev_word, next_word = get_trigram_context(sentence, tagger_for_context)
        if prev_word:
            main_logger.info(f"  (Context for Trigram Filter: ('{prev_word}', BLANK, '{next_word}'))")

        # Loop through all available generators to test each one.
        for name, generator in generators.items():
            if generator is None:
                main_logger.info(f"  - {name}: [Generator not available]")
                continue

            # Generate more candidates than needed (e.g., 20) to give the filters options.
            if name == "Contextual":
                distractor_candidates = generator.generate_distractors(masked_sentence=masked_sentence,
                                                                       target_word=target_word, context_type=context,
                                                                       top_n=20, include_prob_score=False)
            elif name == "Co-occurrence":
                distractor_candidates = generator.generate_distractors(target_word_surface=target_word,
                                                                       sentence_with_blank=sentence,
                                                                       context_type=context, num_distractors=20,
                                                                       include_pmi_score=False)
            elif name == "Similarity":
                distractor_candidates = generator.generate_distractors(target_word=target_word, context_type=context,
                                                                       top_n=20, num_candidates=200,
                                                                       include_sim_score=False)
            else:  # Baseline- and SpellingGenerator
                distractor_candidates = generator.generate_distractors(target_word_surface=target_word,
                                                                       sentence_with_blank=sentence, num_distractors=20)

            # --- Step 4: Apply filters ---
            trigram_accepted_list, trigram_rejected_list = distractor_filter.filter_by_trigram(distractor_candidates,
                                                                                               prev_word, next_word)

            dependency_accepted_list, dependency_rejected_list = distractor_filter.filter_by_dependency(
                distractor_candidates, sentence, )

            bert_accepted_list, bert_rejected_list = distractor_filter.filter_by_bert_fixed(distractor_candidates,
                                                                                            sentence, context,
                                                                                            target_word)

            # Gather all distractors that have been rejected by ANY filter using set union.
            rejected_by_any_filter_set = set(trigram_rejected_list) | set(dependency_rejected_list) | set(
                bert_rejected_list)

            # The final list contains only distractors that were NOT in the combined rejected set.
            final_distractors_list = [
                cand for cand in distractor_candidates if cand not in rejected_by_any_filter_set
            ]

            # Log the final, filtered results.
            main_logger.info(f"\n====== Results for {name}. ======")
            main_logger.info(f"    Distractor candidates pool: {distractor_candidates[:10]}")
            main_logger.info(f"    Trigram Rejects: {list(trigram_rejected_list)[:10]}")
            main_logger.info(f"    Dependency Rejects: {dependency_rejected_list[:10]}")
            main_logger.info(f"    BERT Rejects: {bert_rejected_list[:10]}")
            main_logger.info(f"    Final Rejects (In any filter): {list(rejected_by_any_filter_set)}")
            main_logger.info(f"    Final Distractors: {final_distractors_list[:5]}")

        main_logger.info("-" * 60)

    main_logger.info("\n====== System finished demonstration. ======")
    sys.exit(0)
