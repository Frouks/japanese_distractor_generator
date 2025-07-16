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

    def _calculate_pseudo_log_likelihood(self, sentence: str) -> float:
        """
        Calculates sentence PLL using the PLL-word-l2r method.
        This method corrects for the score inflation of multi-token words by
        also masking subsequent sub-tokens of the same word.
        """
        if not self.bert_model or not self.bert_tokenizer:
            print("Error: BERT model or tokenizer not initialized.")
            return -float('inf')

        tensor_input = self.bert_tokenizer.encode(sentence, return_tensors='pt')

        # Check if the tensor is 2D ([batch_size, sequence_length]), if not we need to add a batch dimension
        if len(tensor_input.shape) == 1:
            tensor_input = tensor_input.unsqueeze(0)

        # To identify which tokens belong to the same word, we need their string representation. -> OOV tokens or complex tokens
        # WordPiece tokenizers (used by BERT) mark sub-tokens of a word with a '##' prefix.
        # Example: "tokenization" -> ["token", "##ization"]
        token_ids = tensor_input[0].tolist()
        tokens = self.bert_tokenizer.convert_ids_to_tokens(token_ids)

        total_log_likelihood = 0.0

        # We iterate through each token in the sentence to calculate its individual PLL.
        # We skip the first token ([CLS]) and the last token ([SEP])
        for i in range(1, tensor_input.shape[1] - 1):
            # Store the original token ID at the current position `i`. We'll need this later
            # to find its log-likelihood in the model's output.
            original_token_id = tensor_input[0, i].item()

            masked_input = tensor_input.clone()

            # 1. Mask the current target token at position `i`.
            # We replace its ID with the special [MASK] token ID.
            masked_input[0, i] = self.bert_tokenizer.mask_token_id

            # 2. Mask all subsequent tokens that are part of the *same word*.
            # We check tokens to the right of the current one (`i + 1`).
            for j in range(i + 1, tensor_input.shape[1] - 1):
                # If a token string starts with '##', it's a continuation of the previous token's word.
                if tokens[j].startswith('##'):
                    # If it's part of the same word, mask it as well.
                    masked_input[0, j] = self.bert_tokenizer.mask_token_id
                else:
                    # If we encounter a token that does NOT start with '##', it means we've
                    # reached the beginning of a new word. We stop masking.
                    break

            # Performance optimization that disables gradient calculation because we are only doing inference, not training.
            with torch.no_grad():
                outputs = self.bert_model(masked_input)

            # [0] -> Selects the results for the first sentence in our batch
            # [i] -> Get predictions for our target position `i`
            logits_for_target_token = outputs.logits[0, i]
            log_probs = torch.nn.functional.log_softmax(logits_for_target_token, dim=0)

            # From the distribution of log probabilities, we select the one corresponding
            # to our *original*, unmasked token. This value is the PLL for this single token.
            token_log_likelihood = log_probs[original_token_id].item()

            # Add this token's score to the running total for the sentence.
            total_log_likelihood += token_log_likelihood

        return total_log_likelihood

    def filter_by_bert(self, candidates: list[str], carrier_sentence: str, context_type: str) -> tuple[
        list[str], list[str]]:
        """
        Filters candidates by rejecting those that make the sentence "too plausible"
        according to BERT's PPL score.
        """
        if not self.bert_model:
            self.logger.warning("BERT model not available. Skipping BERT filtering.")
            return candidates, []
        if "___" not in carrier_sentence:
            self.logger.error("Carrier sentence for BERT filter must contain '___' placeholder.")
            return [], candidates

        # rejection_threshold = -25.0 if context_type == 'closed' else -15.0
        rejection_threshold = -100
        self.logger.info(
            f"--- Filtering with BERT (Context: {context_type}, Rejection Threshold > {rejection_threshold}) ---")

        accepted, rejected = [], []
        for candidate in candidates:
            full_sentence = carrier_sentence.replace("___", candidate)
            score = self._calculate_pseudo_log_likelihood(full_sentence)
            if score <= rejection_threshold:
                self.logger.info(f"  ✅ ACCEPTED: '{candidate}' (Score: {score:.2f})")
                accepted.append(candidate)
            else:
                self.logger.info(f"  ❌ REJECTED: '{candidate}' (Score: {score:.2f})")
                rejected.append(candidate)
        return accepted, rejected


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
                "target": "象", "context": "closed",
                "english_sentence": "At the zoo, the big ___ was raising its trunk high.",
                "prev_word": "大きな", "next_word": "鼻",
                "candidates": ["象", "キリン", "マンモス", "車", "木", "獅子", "アリクイ"],
                "translations": {"象": "elephant", "キリン": "giraffe", "マンモス": "mammoth", "車": "car",
                                 "木": "tree", "獅子": "lion", "アリクイ": "anteater"}
            },
            {
                "sentence": "公園で、たくさんの___が遊んでいた。",
                "target": "子供", "context": "open",
                "english_sentence": "At the park, many ___ were playing.",
                "prev_word": "たくさんの", "next_word": "遊んでいた",
                "candidates": ["子供", "人々", "学生", "動物", "鯉", "成人", "親", "食品"],
                "translations": {"子供": "child", "人々": "people", "学生": "students", "動物": "animals", "鯉": "carp",
                                 "成人": "adults", "親": "parents", "食品": "food"}
            },
            {
                "sentence": "彼は100メートルを10秒で___ことができる。",
                "target": "走る", "context": "closed",
                "english_sentence": "He can ___ 100 meters in 10 seconds.",
                "prev_word": "10秒で", "next_word": "こと",
                "candidates": ["走る", "歩く", "泳ぐ", "飛ぶ", "ある", "飲む", "食べる", "歌う"],
                "translations": {"走る": "run", "歩く": "walk", "泳ぐ": "swim", "飛ぶ": "fly", "ある": "to be",
                                 "飲む": "drink", "食べる": "eat", "歌う": "sing"}
            },
            {
                "sentence": "この___はとても重要です。",
                "target": "問題", "context": "open",
                "english_sentence": "This ___ is very important.",
                "prev_word": "この", "next_word": "とても",
                "candidates": ["問題", "点", "部分", "選手", "脚", "人", "イベント", "食品", "質問", "愛", "歌う",
                               "歌"],
                "translations": {"問題": "problem", "点": "point", "部分": "part", "選手": "athlete", "脚": "leg",
                                 "人": "person", "イベント": "event", "食品": "food", "質問": "question",
                                 "愛": "love", "歌う": "sing", "歌": "song"}
            },
            {
                "sentence": "私の___はとても可愛い。", "target": "猫", "context": "open",
                "english_sentence": "My ___ is very cute.",
                "prev_word": "私の", "next_word": "とても",
                "candidates": ["猫", "犬", "子供", "ハムスター", "カバン"],
                "translations": {"猫": "cat", "犬": "dog", "子供": "child", "ハムスター": "hamster", "カバン": "bag"}
            },
            {
                "sentence": "私の___は可愛くて、よくニャーと鳴く。", "target": "猫", "context": "closed",
                "english_sentence": "My ___ is cute and meows a lot.",
                "prev_word": "私の", "next_word": "可愛くて",
                "candidates": ["猫", "犬", "鳥", "子犬", "子猫"],
                "translations": {"猫": "cat", "犬": "dog", "鳥": "bird", "子犬": "puppy", "子猫": "kitten"}
            }
        ]

        print("=" * 70)

        # --- EXECUTION LOOP ---
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
            # print("\n--- 2. Applying Dependency Filter ---")
            # remaining_after_dep, rejected_by_dep = unified_filter.filter_by_dependency(
            #     remaining_after_trigram, case["sentence"]
            # )
            # print(f"   Rejected {len(rejected_by_dep)} candidates: {format_list(rejected_by_dep)}")
            # print(f"   Candidates remaining: {format_list(remaining_after_dep)}")
            # print("-" * 50)

            # Step 3: BERT Filter
            print("\n--- 3. Applying BERT Filter ---")
            final_distractors, rejected_by_bert = unified_filter.filter_by_bert(
                # remaining_after_dep, case["sentence"], case["context"]
                remaining_after_trigram, case["sentence"], case["context"]
            )
            print("-" * 50)

            # Final Results for this case
            print("\n" + "-" * 25 + f" RESULTS FOR CASE {i + 1} " + "-" * 25)
            print(f"{'Initial Candidates:':<28} {format_list(initial_candidates)}")
            print(f"{'Rejected by Trigram Filter:':<28} {format_list(rejected_by_trigram)}")
            # print(f"{'Rejected by Dependency Filter:':<28} {format_list(rejected_by_dep)}")
            print(f"{'Rejected by BERT Filter:':<28} {format_list(rejected_by_bert)}")
            print("-" * 35)
            print(f"{'Final Accepted Distractors:':<28} {format_list(final_distractors)}")
            print("=" * 70)
