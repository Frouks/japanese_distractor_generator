import logging
import pickle
import time
from pathlib import Path

import spacy
import torch
from transformers import BertTokenizer, BertForMaskedLM


class DistractorFilter:
    """
    A unified filter that can screen distractors using trigrams, dependency relations,
    and BERT-based sentence plausibility (Pseudo-Log-Likelihood).
    """

    def __init__(self, trigram_path, dependency_index_path,
                 bert_model_name='cl-tohoku/bert-base-japanese-whole-word-masking'):
        """
        Initializes the filter by loading all required data sets and models.
        """
        self.logger = logging.getLogger('DistractorFilter')

        # --- Load Trigram and Dependency Data ---
        self.trigrams = self._load_pickle_data(
            Path(trigram_path),
            "trigrams",
            empty_type=set()
        )

        self.dependency_index = self._load_pickle_data(
            Path(dependency_index_path),
            "dependency index",
            empty_type={}
        )
        # --- Load spaCy/GiNZA for Dependency Parsing ---
        self.logger.info("Loading spaCy/GiNZA model for live parsing...")
        try:
            self.nlp = spacy.load("ja_ginza")
            self.logger.info("✅ GiNZA model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load GiNZA model: {e}", exc_info=True)
            self.nlp = None

        # --- Load BERT Model for PPL Filtering ---
        self.logger.info(f"Loading BERT model '{bert_model_name}' for filtering...")
        try:
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
            self.bert_model = BertForMaskedLM.from_pretrained(bert_model_name)
            self.bert_model.eval()
            self.logger.info("✅ BERT model loaded successfully for filtering.")
        except Exception as e:
            self.logger.error(f"Failed to load BERT model: {e}", exc_info=True)
            self.bert_tokenizer = None
            self.bert_model = None

    def _load_pickle_data(self, file_path, data_name, empty_type):
        """
        Helper method to load a pickled data object from a file.
        Returns a given empty object on failure.

        Args:
            file_path (Path): Path to the pickle file.
            data_name (str): Name for logging.
            empty_type: The empty object to return on failure (e.g., set(), {}).
        """
        if not file_path.exists():
            self.logger.error(f"{data_name.capitalize()} data file not found: {file_path}")
            return empty_type

        self.logger.info(f"Loading {data_name} from {file_path}...")
        start_time = time.time()
        try:
            with open(file_path, 'rb') as f:
                data_obj = pickle.load(f)
            duration = time.time() - start_time
            self.logger.info(
                f"Successfully loaded {len(data_obj):,} items into a {type(data_obj).__name__} in {duration:.2f}s.")
            return data_obj
        except Exception as e:
            self.logger.error(f"Failed to load {data_name} data: {e}", exc_info=True)
            return empty_type

    def filter_by_trigram(self, candidates: list[str], prev_word: str, next_word: str) -> tuple[list[str], list[str]]:
        if not self.trigrams:
            self.logger.warning("Trigram set is empty. Skipping trigram filtering.")
            return candidates, []

        accepted, rejected = [], []
        for candidate in candidates:
            trigram_to_check = (prev_word, candidate, next_word)
            if trigram_to_check not in self.trigrams:
                accepted.append(candidate)
            else:
                rejected.append(candidate)

        return accepted, rejected

    def filter_by_dependency(self, candidates: list[str], sentence_template: str, blank_placeholder: str = "___") -> \
            tuple[list[str], list[str]]:
        if not self.dependency_index or not self.nlp:
            self.logger.warning("Dependency index or GiNZA model not available. Skipping dependency filtering.")
            return candidates, []

        accepted, rejected = [], []
        for candidate in candidates:
            full_sentence = sentence_template.replace(blank_placeholder, candidate, 1)
            is_rejected = False
            try:
                doc = self.nlp(full_sentence)
                for token in doc:
                    # We only care about relations where the candidate is the head or the child
                    if token.lemma_ != candidate and token.head.lemma_ != candidate:
                        continue

                    live_relation = (token.dep_, token.head.lemma_, token.lemma_)
                    # Efficiently check if the head of the relation is in our index
                    known_relations_for_head = self.dependency_index.get(live_relation[1], [])

                    if live_relation in known_relations_for_head:
                        is_rejected = True
                        rejected.append(candidate)
                        break  # Found a reason to reject, move to next candidate

                if not is_rejected:
                    accepted.append(candidate)

            except Exception as e:
                self.logger.error(f"Error during dependency parsing for candidate '{candidate}': {e}")
                accepted.append(candidate)  # Failsafe: accept if parsing fails

        return accepted, rejected

    def _calculate_pll(self, sentence: str) -> float:
        """
        Calculates the sentence PLL using the PLL-word-l2r method.
        This corrects for score inflation from multi-token words.
        """
        if not self.bert_model or not self.bert_tokenizer:
            self.logger.error("BERT model or tokenizer not initialized.")
            return -float('inf')

        # Tokenize the input sentence to get token IDs and string representations
        token_ids = self.bert_tokenizer.encode(sentence, add_special_tokens=False)
        tokens = self.bert_tokenizer.convert_ids_to_tokens(token_ids)

        # Add [CLS] and [SEP] special tokens for BERT processing
        input_ids = self.bert_tokenizer.build_inputs_with_special_tokens(token_ids)
        tensor_input = torch.tensor([input_ids])

        if tensor_input.shape[1] <= 2:  # Only [CLS] and [SEP]
            return -float('inf')

        total_log_likelihood = 0.0

        # Iterate through each actual token (skipping [CLS] and [SEP])
        for i in range(1, len(tokens) + 1):
            token_of_interest = tensor_input[0, i].item()
            masked_input = tensor_input.clone()

            # --- PLL-word-l2r Logic ---
            # 1. Mask the current target token
            masked_input[0, i] = self.bert_tokenizer.mask_token_id

            # 2. Also mask subsequent tokens if they are part of the same word
            for j in range(i + 1, len(tokens) + 1):
                if tokens[j - 1].startswith('##'):
                    masked_input[0, j] = self.bert_tokenizer.mask_token_id
                else:
                    break  # Stop masking at the start of a new word

            # Get model predictions with no gradient calculation for efficiency
            with torch.no_grad():
                outputs = self.bert_model(masked_input)

            # Extract the log probability of the token_of_interest -> the one originally "below" the mask
            logits = outputs.logits[0, i]
            log_probs = torch.nn.functional.log_softmax(logits, dim=0)
            token_log_likelihood = log_probs[token_of_interest].item()
            total_log_likelihood += token_log_likelihood

        # num_tokens = len(tokens)
        # normalized_pll = total_log_likelihood / num_tokens if num_tokens > 0 else 0.0

        return total_log_likelihood

    def filter_by_bert_fixed(self, candidates: list[str], carrier_sentence: str, context: str, target_word: str) -> \
            tuple[
                list[str], list[str]]:
        """
        Filters candidates using a PLL score. Candidates are rejected
        if they form a sentence that is too plausible (i.e., the score
        is above a fixed threshold).
        """
        if not self.bert_model:
            self.logger.warning("BERT model not available. Skipping BERT filtering.")
            return candidates, []
        if "___" not in carrier_sentence:
            self.logger.error("Carrier sentence for BERT filter must contain '___' placeholder.")
            return [], candidates

        accepted, rejected = [], []
        sentence_with_target_word = carrier_sentence.replace("___", target_word)
        target_word_pll = self._calculate_pll(sentence_with_target_word)

        if context == "Open":
            pll_threshold = target_word_pll - 1.25
        else:
            pll_threshold = target_word_pll - 0.5

        self.logger.info(f"Threshold: > {pll_threshold}")
        for candidate in candidates:
            full_sentence = carrier_sentence.replace("___", candidate)
            score = self._calculate_pll(full_sentence)

            if score <= pll_threshold:
                self.logger.info(f"  ✅ ACCEPTED: '{candidate}' (PLL Score: {score:.2f})")
                accepted.append(candidate)
            else:
                self.logger.info(f"  ❌ REJECTED: '{candidate}' (PLL Score: {score:.2f})")
                rejected.append(candidate)
        return accepted, rejected

    # def filter_by_bert_percentile(self, candidates: list[str], carrier_sentence: str, target_word: str,
    #                               percentile: int = 25) -> tuple[list[str], list[str]]:
    #     """Filters candidates based on the percentile rank of their PLL scores."""
    #     if not self.bert_model or not candidates: return candidates, []
    #
    #     # Calculate scores for all candidates and the target word
    #     candidate_scores = {c: self._calculate_normalized_pll(carrier_sentence.replace("___", c)) for c in candidates}
    #     target_score = self._calculate_normalized_pll(carrier_sentence.replace("___", target_word))
    #
    #     all_scores = list(candidate_scores.values()) + [target_score]
    #
    #     # Determine the threshold from the distribution of scores
    #     threshold = np.percentile(all_scores, percentile)
    #     self.logger.info(
    #         f"--- Filtering with BERT (Percentile Threshold @ {percentile}%: < {threshold:.2f}) ---")
    #
    #     accepted, rejected = [], []
    #     for candidate, score in candidate_scores.items():
    #         if score >= threshold and candidate != target_word:
    #             self.logger.info(f"  ✅ ACCEPTED: '{candidate}' (Score: {score:.2f})")
    #             accepted.append(candidate)
    #         else:
    #             self.logger.info(f"  ❌ REJECTED: '{candidate}' (Score: {score:.2f})")
    #             rejected.append(candidate)
    #     return accepted, rejected


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Initializing the Unified DistractorFilter...")
    DATA_DIR = Path("../../re_implementation/processed_corpus_data")
    TRIGRAM_FILE = DATA_DIR / "jp_trigram_counts.pkl"
    DEPENDENCY_FILE = DATA_DIR / "jp_dependency_relations.pkl"

    unified_filter = DistractorFilter(TRIGRAM_FILE, DEPENDENCY_FILE)
    print("=" * 60)

    if unified_filter.bert_model:
        # --- SETUP: TEST CASES AND TRANSLATIONS ---
        test_cases: list[dict[str, any]] = [
            {
                "sentence": "動物園で、大きな___が鼻を高く上げていた。",
                "target": "象", "context": "Closed",
                "english_sentence": "At the zoo, the big ___ was raising its trunk high.",
                "prev_word": "大きな", "next_word": "鼻",
                "candidates": ["象", "キリン", "マンモス", "車", "木", "獅子", "アリクイ"],
                "translations": {"象": "elephant", "キリン": "giraffe", "マンモス": "mammoth", "車": "car",
                                 "木": "tree", "獅子": "lion", "アリクイ": "anteater"}
            },
            {
                "sentence": "公園で、たくさんの___が遊んでいた。",
                "target": "子供", "context": "Open",
                "english_sentence": "At the park, many ___ were playing.",
                "prev_word": "たくさんの", "next_word": "遊んでいた",
                "candidates": ["子供", "人々", "学生", "動物", "鯉", "成人", "親", "食品"],
                "translations": {"子供": "child", "人々": "people", "学生": "students", "動物": "animals", "鯉": "carp",
                                 "成人": "adults", "親": "parents", "食品": "food"}
            },
            {
                "sentence": "彼は100メートルを10秒で___ことができる。",
                "target": "走る", "context": "Closed",
                "english_sentence": "He can ___ 100 meters in 10 seconds.",
                "prev_word": "10秒で", "next_word": "こと",
                "candidates": ["走る", "歩く", "泳ぐ", "飛ぶ", "ある", "飲む", "食べる", "歌う"],
                "translations": {"走る": "run", "歩く": "walk", "泳ぐ": "swim", "飛ぶ": "fly", "ある": "to be",
                                 "飲む": "drink", "食べる": "eat", "歌う": "sing"}
            },
            {
                "sentence": "この___はとても重要です。",
                "target": "問題", "context": "Open",
                "english_sentence": "This ___ is very important.",
                "prev_word": "この", "next_word": "とても",
                "candidates": ["問題", "点", "部分", "選手", "脚", "人", "イベント", "食品", "質問", "愛", "歌う",
                               "歌"],
                "translations": {"問題": "problem", "点": "point", "部分": "part", "選手": "athlete", "脚": "leg",
                                 "人": "person", "イベント": "event", "食品": "food", "質問": "question",
                                 "愛": "love", "歌う": "sing", "歌": "song"}
            },
            {
                "sentence": "私の___はとても可愛い。", "target": "猫", "context": "Open",
                "english_sentence": "My ___ is very cute.",
                "prev_word": "私の", "next_word": "とても",
                "candidates": ["猫", "犬", "子供", "ハムスター", "カバン"],
                "translations": {"猫": "cat", "犬": "dog", "子供": "child", "ハムスター": "hamster", "カバン": "bag"}
            },
            {
                "sentence": "私の___は可愛くて、よくニャーと鳴く。", "target": "猫", "context": "Closed",
                "english_sentence": "My ___ is cute and meows a lot.",
                "prev_word": "私の", "next_word": "可愛くて",
                "candidates": ["猫", "犬", "鳥", "子犬", "子猫"],
                "translations": {"猫": "cat", "犬": "dog", "鳥": "bird", "子犬": "puppy", "子猫": "kitten"}
            }
        ]

        print("=" * 70)

        for i, case in enumerate(test_cases):
            english_sentence_with_blank = case["english_sentence"]

            print(f"🧪 TEST CASE {i + 1}: {case['context'].upper()} CONTEXT")
            print(f"   Sentence: {case['sentence']}")
            print(f"   English:  {english_sentence_with_blank}'")

            initial_candidates = case["candidates"]
            translations = case["translations"]


            def format_list(candidate_list):
                return [f"{c} ({translations.get(c, 'N/A')})" for c in candidate_list]


            print(f"   Initial Candidates: {format_list(initial_candidates)}")
            print("=" * 70)

            # Step 1: Trigram Filter
            print("\n--- 1. Applying Trigram Filter ---")
            remaining_after_trigram, rejected_by_trigram = unified_filter.filter_by_trigram(
                initial_candidates, case["prev_word"], case["next_word"]
            )
            print(f"   Rejected {len(rejected_by_trigram)} candidates: {format_list(rejected_by_trigram)}")
            print(f"   Candidates remaining: {format_list(remaining_after_trigram)}")
            print("-" * 50)

            # Step 2: Dependency Filter
            print("\n--- 2. Applying Dependency Filter ---")
            remaining_after_dep, rejected_by_dep = unified_filter.filter_by_dependency(
                remaining_after_trigram, case["sentence"]
            )
            print(f"   Rejected {len(rejected_by_dep)} candidates: {format_list(rejected_by_dep)}")
            print(f"   Candidates remaining: {format_list(remaining_after_dep)}")
            print("-" * 50)

            # Step 3: BERT Filter
            print("\n--- 3. Applying BERT Filter ---")
            final_distractors, rejected_by_bert = unified_filter.filter_by_bert_fixed(
                remaining_after_dep, case["sentence"], case["context"], case["target"]
                # remaining_after_trigram, case["sentence"], case["context"], case["target"]
            )
            print("-" * 50)

            # Final Results for this case
            print("\n" + "-" * 25 + f" RESULTS FOR CASE {i + 1} " + "-" * 25)
            print(f"{'Initial Candidates:':<28} {format_list(initial_candidates)}")
            print(f"{'Rejected by Trigram Filter:':<28} {format_list(rejected_by_trigram)}")
            print(f"{'Rejected by Dependency Filter:':<28} {format_list(rejected_by_dep)}")
            print(f"{'Rejected by BERT Filter:':<28} {format_list(rejected_by_bert)}")
            print("-" * 35)
            print(f"{'Final Accepted Distractors:':<28} {format_list(final_distractors)}")
            print("=" * 70)
