import json
import logging
from collections import Counter
from pathlib import Path

import MeCab
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# --- Notes ---
# What is a lemma?
# The lemma (or base form) is the dictionary form of a word.
# e.g., for "running", "ran", "runs" -> the lemma is "run".
# e.g., for 食べた (ate), 食べない (don't eat) -> the lemma is 食べる (to eat).
# We use lemmas to group all variations of a word, giving a more accurate frequency count for its concept.

# What is the difference between the surfaceForm and baseForm?
# surfaceForm: The word exactly as it appeared in the text. (食べた - "ate")
# baseForm (Lemma): The dictionary form of that word. (食べる - "to eat")
class CorpusProcessor:
    """
    This class is responsible for processing a large text corpus of Japanese sentences.
    Its main goals are:
    1. To tokenize sentences into words (lemmas).
    2. To identify the Part-of-Speech (POS) for each word.
    3. To count the frequency of each unique (lemma, POS) pair in the corpus.
    4. To save this processed information for later use in distractor generation.
    """

    def __init__(self, mecab_args=''):
        """
        Initializes the CorpusProcessor.
        - Sets up logging.
        - Initializes the MeCab tagger with optional arguments.
        - Defines critical indices for parsing MeCab's output.
        - Initializes data structures to store corpus statistics.
        
        Args:
            mecab_args (str): A string of arguments to pass to the MeCab Tagger.
                              This allows for specifying a particular dictionary (e.g., '-d /path/to/dic').
                              If empty, MeCab uses its system default.
        """
        self.logger = logging.getLogger('CorpusProcessor')
        self.mecab_args = mecab_args

        if not self.mecab_args:
            self.logger.info(
                "No explicit MeCab args provided by user. MeCab will use its default dictionary configuration.")
            self.logger.warning(
                "IMPORTANT: Please check the 'Active MeCab Dictionary Info' log output below to ensure MeCab is using the dictionary you expect (e.g., unidic).")

        try:
            self.tagger = MeCab.Tagger(self.mecab_args)
            self.logger.info(f"MeCab Tagger initialized with final effective args: '{self.mecab_args}'.")

            # Get information about the dictionary MeCab actually loaded
            info_node = self.tagger.dictionary_info()
            if info_node:
                dict_info_str = "Active MeCab Dictionary Info (from Tagger):\n"
                while info_node:
                    dict_info_str += (f"  Filename: {info_node.filename}, Charset: {info_node.charset}, "
                                      f"Type: {info_node.type}, Version: {info_node.version}\n")
                    info_node = info_node.next
                self.logger.info(dict_info_str)
            else:
                self.logger.warning(
                    "Could not retrieve MeCab dictionary info (tagger.dictionary_info() returned None). This might occur if MeCab couldn't load any dictionary.")

        except RuntimeError as e:
            self.logger.error(
                f"CRITICAL: Failed to initialize MeCab Tagger with effective args '{self.mecab_args}': {e}",
                exc_info=True)
            self.logger.error("Ensure MeCab (the C++ library) is installed correctly and a dictionary is accessible "
                              "either via the provided args or as a system default recognized by MeCab.")
            raise

        # MeCab constants for node status (BOS/EOS)
        try:
            self.MECAB_BOS_NODE = MeCab.MECAB_BOS_NODE
            self.MECAB_EOS_NODE = MeCab.MECAB_EOS_NODE
        except AttributeError:
            self.logger.error(
                "MeCab.MECAB_BOS_NODE or MeCab.MECAB_EOS_NODE not found. This mecab-python3 version might be too old or the MeCab module is not fully loaded.",
                exc_info=True)
            self.MECAB_BOS_NODE = 1
            self.MECAB_EOS_NODE = 2
            self.logger.warning(
                f"Falling back to assumed integer values for BOS/EOS nodes: BOS={self.MECAB_BOS_NODE}, EOS={self.MECAB_EOS_NODE}. VERIFY THIS if parsing issues occur.")

        self.idx_pos1 = 0  # Index for the main POS category (e.g., 名詞)
        self.idx_pos_detail_end = 3  # Index for the last part of detailed POS (e.g., features[0] to features[3])
        self.idx_lemma = 7  # Index for the Lemma/Base form (e.g., 食べる)
        self.logger.info(
            f"Using feature indices for parsing: POS1={self.idx_pos1}, POS_Detail_End={self.idx_pos_detail_end}, Lemma={self.idx_lemma}")

        # Instance variable initializations
        self.word_pos_frequencies = Counter()  # To count (lemma, POS_major) frequencies.
        self.all_words_details = {}  # To store detailed info for each unique (lemma, POS_major).
        self.unknown_word_log = []  # To log examples of words the dictionary couldn't fully analyze.
        self.suspicious_pos_log = []  # For logging words with potentially incorrect POS tags (not actively used yet).
        self.total_tokens_processed = 0  # Counter for all MeCab nodes yielded (including BOS/EOS).
        self.unknown_token_count = 0  # Counter for tokens where the dictionary indicates the lemma is unknown ('*').

    def parse_mecab_node(self, node, current_sentence_text=""):
        """
        Parses a single MeCab node to extract structured token information.
        - Skips irrelevant nodes (BOS/EOS).
        - Extracts surface form, base form (lemma), and POS.
        - Handles unknown words.
        - Filters out non-content words like particles, symbols and suffixes.
        
        Args:
            node (MeCab.Node): A MeCab node object from `tagger.parseToNode()`.
            current_sentence_text (str): The sentence being parsed, for logging context.

        Returns:
            dict or None: A dictionary with token info, or None if the node should be skipped.
        """
        surface = node.surface

        # Skip if the node is BOS/EOS or if the surface form is empty/whitespace.
        if not surface.strip() or node.stat == self.MECAB_BOS_NODE or node.stat == self.MECAB_EOS_NODE:
            return None

        features = node.feature.split(',')

        required_len = max(self.idx_pos1, self.idx_pos_detail_end, self.idx_lemma) + 1
        if len(features) < required_len:
            self.logger.debug(
                f"Skipping node: Insufficient features. Surface='{surface}', Features='{node.feature}', Context='{current_sentence_text[:30]}...'")
            return None

        pos_major = features[self.idx_pos1]
        pos_detailed_str = ",".join(features[self.idx_pos1: self.idx_pos_detail_end + 1])

        excluded_pos_major = [
            '補助記号',  # Supplementary Symbol (e.g., commas, periods)
            '助詞',  # Particle (e.g., は, が, を - grammatical markers)
            '助動詞',  # Auxiliary Verb (e.g., ます, です, ない - grammatical endings)
            '接尾辞',  # Suffix (e.g., -的, -化, -者)
            '感動詞',  # Interjection (e.g., "Oh!", "Wow!")
            'フィラー',  # Filler (e.g., "um", "uh")
            'その他',  # Other
            '空白',  # Whitespace
            '記号',  # Symbol (general category)
            '接頭辞',  # Prefix (e.g., 御-, 無-, 非-)
        ]

        if pos_major in excluded_pos_major:
            return None

        lemma_from_features = features[self.idx_lemma]
        base_form = lemma_from_features if lemma_from_features.strip() and lemma_from_features != '*' else surface
        is_unknown_by_dict = (lemma_from_features == '*')

        if is_unknown_by_dict:
            self.unknown_token_count += 1
            if len(self.unknown_word_log) < 1000:
                self.unknown_word_log.append({
                    'surface': surface, 'features': node.feature,
                    'context': current_sentence_text[:50] + "..."})

        if not base_form.strip():
            self.logger.debug(
                f"Skipping node with empty base_form after processing: Surface='{surface}', Lemma from features='{lemma_from_features}'")
            return None

        return {
            'surface': surface, 'base_form': base_form, 'pos_major': pos_major,
            'pos_detailed': pos_detailed_str, 'is_unknown_by_dict': is_unknown_by_dict
        }

    def process_sentences_from_iterable(self, sentence_iterable, total_sentences=None):
        """
        The main processing loop.
        - Iterates through sentences, tokenizes each one, and aggregates word frequency and details.
        - The look-ahead logic is restricted to verbs and adjectives being followed by 
          auxiliary verbs, preventing incorrect noun-suffix combination.
          
        Args:
            sentence_iterable (iterable): An iterable that yields sentences (strings).
            total_sentences (int, optional): The total number of sentences for tqdm's progress bar.
        """
        self.logger.info("Starting MeCab processing from sentence iterable...")
        self.total_tokens_processed = 0
        self.unknown_token_count = 0
        self.word_pos_frequencies.clear()
        self.all_words_details.clear()
        self.unknown_word_log.clear()

        # POS tags that can be a base for reconstruction.
        reconstruct_base_pos = {'動詞', '形容詞'}  # 動詞 (Verb), 形容詞 (Adjective)
        # POS tags that can be attached to verbs/adjectives.
        attachable_pos_for_verbs_adj = {'助動詞'}  # 助動詞 (Auxiliary Verb)

        progress_bar = tqdm(sentence_iterable, total=total_sentences, desc="Tokenizing with MeCab")
        # Loop through each sentence provided by the iterable -> we request the next sentence when we are done with the current one
        for sentence_text in progress_bar:
            if not isinstance(sentence_text, str) or not sentence_text.strip():
                continue

            try:
                self.tagger.parse('')
                node = self.tagger.parseToNode(sentence_text)

                while node:
                    self.total_tokens_processed += 1

                    # Parse the current node to see if it's a content word.
                    token_info = self.parse_mecab_node(node, sentence_text)

                    if token_info:
                        # Check if this token is a verb or adjective, which might need reconstruction.
                        if token_info['pos_major'] in reconstruct_base_pos:
                            combined_surface = token_info['surface']
                            lookahead_node = node.next

                            while lookahead_node:
                                if not lookahead_node.surface.strip() or lookahead_node.stat in (self.MECAB_BOS_NODE,
                                                                                                 self.MECAB_EOS_NODE):
                                    break

                                features = lookahead_node.feature.split(',')

                                # Check if the lookahead node is an attachable auxiliary verb.
                                if len(features) > self.idx_pos1 and features[
                                    self.idx_pos1] in attachable_pos_for_verbs_adj:
                                    combined_surface += lookahead_node.surface
                                    node = lookahead_node  # Consume the node
                                    lookahead_node = node.next
                                else:
                                    break  # Stop looking ahead.

                            # Update the surface form with the full, combined version.
                            token_info['surface'] = combined_surface

                        # Store the final token info. For nouns, this will be the original, unmodified token.
                        # For verbs/adjectives, it will be the reconstructed one.
                        # Create the key for our dictionaries: a tuple of (lemma, POS).
                        key = (token_info['base_form'], token_info['pos_major'])
                        self.word_pos_frequencies[key] += 1

                        if key not in self.all_words_details:
                            self.all_words_details[key] = {
                                'surface_forms': set(),
                                'detailed_pos_tags': set(),
                                'frequency': 0,
                                'unknown_by_dict_count': 0
                            }

                        self.all_words_details[key]['surface_forms'].add(token_info['surface'])
                        self.all_words_details[key]['detailed_pos_tags'].add(token_info['pos_detailed'])
                        if token_info['is_unknown_by_dict']:
                            self.all_words_details[key]['unknown_by_dict_count'] += 1

                    # Move to the next node in the original sequence.
                    node = node.next
            except Exception as e:
                self.logger.error(f"Error tokenizing sentence: '{sentence_text[:70]}...': {e}", exc_info=True)

        # After processing, update frequencies in all_words_details
        for key, freq in self.word_pos_frequencies.items():
            if key in self.all_words_details:
                self.all_words_details[key]['frequency'] = freq

        self.logger.info("MeCab sentence processing finished.")
        self.logger.info(f"Total MeCab nodes yielded (incl. BOS/EOS): {self.total_tokens_processed:,}")

        num_content_tokens_processed = sum(self.word_pos_frequencies.values())
        self.logger.info(f"Total content tokens (after filtering and combining): {num_content_tokens_processed:,}")

        if num_content_tokens_processed > 0:
            unknown_percentage = (self.unknown_token_count / num_content_tokens_processed) * 100 \
                if self.unknown_token_count <= num_content_tokens_processed else \
                (self.unknown_token_count / self.total_tokens_processed) * 100  # Fallback
            self.logger.info(
                f"Content tokens where dictionary lemma was '*' (is_unknown_by_dict=True): {self.unknown_token_count:,} ({unknown_percentage:.2f}% of content tokens).")
        elif self.total_tokens_processed > 0:
            unknown_percentage_of_total = (self.unknown_token_count / self.total_tokens_processed) * 100
            self.logger.info(
                f"Content tokens where dictionary lemma was '*' (is_unknown_by_dict=True): {self.unknown_token_count:,} ({unknown_percentage_of_total:.2f}% of total MeCab nodes).")
        else:
            self.logger.info(
                f"Content tokens where dictionary lemma was '*' (is_unknown_by_dict=True): {self.unknown_token_count:,} (N/A %)")

        self.logger.info(f"Unique (base_form, pos_major) types found: {len(self.word_pos_frequencies):,}")
        if self.unknown_word_log:
            self.logger.info(
                f"Logged {len(self.unknown_word_log)} examples of words where dictionary lemma was '*'. Review logs or saved JSON.")

    def get_token_info_for_word(self, word_surface):
        """
        IMPORTANT: If we have the sentence it is recommended to use 'get_target_info_in_context' for better result.
        A utility to parse a single, isolated word. 
        This is useful for getting the lemma and POS of a target word for distractor generation.
        
        Args:
            word_surface (str): The surface form of the word to analyze.

        Returns:
            dict or None: A dictionary with the token's info, or None if it cannot be parsed.
        """
        if not isinstance(word_surface, str) or not word_surface.strip():
            self.logger.warning("get_token_info_for_word received empty or non-string input.")
            return None

        self.tagger.parse('')
        node = self.tagger.parseToNode(word_surface)

        while node:
            if node.surface.strip() and not (node.stat == self.MECAB_BOS_NODE or node.stat == self.MECAB_EOS_NODE):
                current_token_info = self.parse_mecab_node(node, word_surface)
                if current_token_info:
                    return current_token_info
            node = node.next

        # self.logger.warning(f"Could not reliably parse or determine primary token info for input: '{word_surface}' using MeCab.")
        return None

    def get_target_info_in_context(self, sentence_with_blank, target_word_surface, placeholder="___"):
        """
        Analyzes the target word within its full sentence context.
        This is the recommended method for getting target word info.
        """
        if not all([sentence_with_blank, target_word_surface]):
            self.logger.error("get_target_info_in_context received empty sentence or target word.")
            return None

        # Step 1: Reconstruct the full sentence
        full_sentence = sentence_with_blank.replace(placeholder, target_word_surface, 1)

        # Step 2: Tokenize using the same advanced look-ahead logic as the main processor
        all_tokens_in_sentence = []
        try:
            self.tagger.parse('')
            node = self.tagger.parseToNode(full_sentence)

            while node:
                self.total_tokens_processed += 1
                token_info = self.parse_mecab_node(node, full_sentence)

                if token_info:
                    # Apply the same look-ahead logic for verbs/adjectives
                    if token_info['pos_major'] in {'動詞', '形容詞'}:
                        combined_surface = token_info['surface']
                        lookahead_node = node.next

                        while lookahead_node:
                            if not lookahead_node.surface.strip() or lookahead_node.stat in (self.MECAB_BOS_NODE,
                                                                                             self.MECAB_EOS_NODE):
                                break

                            features = lookahead_node.feature.split(',')
                            if len(features) > self.idx_pos1 and features[self.idx_pos1] in {'助動詞'}:
                                combined_surface += lookahead_node.surface
                                node = lookahead_node
                                lookahead_node = node.next
                            else:
                                break

                        token_info['surface'] = combined_surface

                    all_tokens_in_sentence.append(token_info)

                node = node.next
        except Exception as e:
            self.logger.error(f"Error tokenizing context sentence: '{full_sentence[:70]}...': {e}", exc_info=True)
            return None

        # Step 3: Find the token that matches our target surface form
        # This handles cases where the target word might appear multiple times.
        # We find the instance that was at the placeholder's location.
        pre_blank_text = sentence_with_blank.split(placeholder, 1)[0]
        # Count the number of tokens *before* the blank.
        # We need a temporary parse of the pre-blank text. This is quick.
        pre_blank_tokens_count = 0
        node = self.tagger.parseToNode(pre_blank_text)
        while node:
            if self.parse_mecab_node(node):  # This re-uses the filtering logic
                pre_blank_tokens_count += 1
            node = node.next

        # The target token should be at this index in our list of all_tokens_in_sentence
        if pre_blank_tokens_count < len(all_tokens_in_sentence):
            target_token_info = all_tokens_in_sentence[pre_blank_tokens_count]
            # Final sanity check
            if target_word_surface in target_token_info['surface']:
                return target_token_info

        # Fallback search if index logic fails (should be rare)
        for token in all_tokens_in_sentence:
            if token['surface'] == target_word_surface:
                self.logger.warning("Used fallback search to find target token in context.")
                return token

        self.logger.error(
            f"Could not locate the target word '{target_word_surface}' within the context of the parsed sentence.")
        return None

    def save_processed_data(self, output_dir="processed_corpus_data",
                            details_filename="jp_all_words_details.json",
                            unknown_log_filename="unknown_words_sample.json"):
        """
        Saves the processed corpus data to JSON files for later use.

        Args:
            output_dir (str): The directory where files will be saved.
            details_filename (str): Filename for the main word details data.
            unknown_log_filename (str): Filename for the sample of unknown words.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        serializable_words_info = {}
        for key_tuple, data in self.all_words_details.items():
            str_key = f"{key_tuple[0]}|{key_tuple[1]}"
            data_copy = data.copy()
            data_copy['surface_forms'] = sorted(list(data_copy['surface_forms']))
            data_copy['detailed_pos_tags'] = sorted(list(data_copy['detailed_pos_tags']))
            serializable_words_info[str_key] = data_copy

        details_file_path = output_path / details_filename
        try:
            with open(details_file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_words_info, f, ensure_ascii=False, indent=2)
            self.logger.info(f"All words details with frequencies saved to {details_file_path.resolve()}")
        except Exception as e:
            self.logger.error(f"Failed to save words details to {details_file_path}: {e}", exc_info=True)

        if self.unknown_word_log:
            unknown_log_file_path = output_path / unknown_log_filename
            try:
                with open(unknown_log_file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.unknown_word_log, f, ensure_ascii=False, indent=2)
                self.logger.info(
                    f"Sample of words with dictionary lemma='*' logged to {unknown_log_file_path.resolve()}")
            except Exception as e:
                self.logger.error(f"Failed to save unknown words log to {unknown_log_file_path}: {e}", exc_info=True)

    @staticmethod
    def load_all_words_details(details_file_path_str="processed_corpus_data/jp_all_words_details.json"):
        """
        A static method to load the processed word details from a JSON file.

        Args:
            details_file_path_str (str): The path to the JSON file to load.

        Returns:
            dict or None: A dictionary with the loaded data, with keys converted back to tuples.
        """
        logger = logging.getLogger('CorpusProcessor.Loader')
        details_file_path = Path(details_file_path_str)

        if not details_file_path.exists():
            logger.error(f"Words details file not found: {details_file_path.resolve()}")
            return None

        logger.info(f"Loading all words details from {details_file_path.resolve()}...")
        try:
            with open(details_file_path, 'r', encoding='utf-8') as f:
                serializable_words_info = json.load(f)

            loaded_all_words_details = {}
            for str_key, data in serializable_words_info.items():
                parts = str_key.split('|', 1)
                if len(parts) == 2:
                    loaded_all_words_details[(parts[0], parts[1])] = data
                else:
                    logger.warning(f"Skipping malformed key in details file: '{str_key}'")

            logger.info(f"Successfully loaded {len(loaded_all_words_details)} entries from words details file.")
            return loaded_all_words_details
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error loading words details from {details_file_path}: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Error loading words details from {details_file_path}: {e}", exc_info=True)
            return None

    def generate_corpus_stats_visualizations(self, output_dir="figures", top_n=20):
        """
        Generates and saves plots to visualize the statistics of the processed corpus.
        
        Args:
            output_dir (str): The directory where the plots will be saved.
            top_n (int): The number of top items to display in the plots.
        """
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Generating corpus statistics visualizations in {output_dir_path.resolve()}...")

        if not self.word_pos_frequencies:
            self.logger.warning("No processed word frequency data available. Skipping visualization.")
            return

        pos_translation_dict = {
            '名詞': 'Noun', '動詞': 'Verb', '形容詞': 'Adjective', '副詞': 'Adverb',
            '連体詞': 'Adnominal', '接続詞': 'Conjunction', '代名詞': 'Pronoun',
            '形状詞': 'Adjectival Noun', '接尾辞': 'Suffix', '感動詞': 'Interjection',
            '助動詞': 'Auxiliary Verb',
        }

        try:
            plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'MS Gothic', 'Noto Sans CJK JP',
                                               'sans-serif']
        except Exception:
            self.logger.warning("Could not set a specific Japanese font list for plots.")

        # Plot 1: Top N Most Frequent Words
        fig1, ax1 = plt.subplots(figsize=(12, max(8, top_n * 0.45)))
        top_n_items = self.word_pos_frequencies.most_common(top_n)
        if top_n_items:
            labels = [f"{item[0][0]} ({pos_translation_dict.get(item[0][1], item[0][1])})" for item in top_n_items]
            counts = [item[1] for item in top_n_items]
            sns.barplot(x=counts, y=labels, palette="viridis", hue=labels, ax=ax1, legend=False)
            ax1.set_title(f'Top {len(labels)} Most Frequent Vocabulary Items (Lemma)', fontsize=15)
            ax1.set_xlabel('Frequency in Corpus', fontsize=12)
            ax1.set_ylabel('Word (POS)', fontsize=12)
            plt.tight_layout()
            try:
                fig1.savefig(output_dir_path / "top_n_word_pos_frequency.png", dpi=300)
                self.logger.info(f"Saved top_n_word_pos_frequency.png")
            except Exception as e:
                self.logger.error(f"Failed to save top_n_word_pos_frequency.png: {e}", exc_info=True)
        plt.close(fig1)

        # Plot 2: Distribution of Major POS Tags
        pos_major_token_sum_counts = Counter()
        for (lemma, pos_major), freq in self.word_pos_frequencies.items():
            pos_major_token_sum_counts[pos_major] += freq
        if pos_major_token_sum_counts:
            fig2, ax2 = plt.subplots(figsize=(14, max(8, len(pos_major_token_sum_counts) * 0.4)))
            sorted_pos = sorted(pos_major_token_sum_counts.items(), key=lambda item: item[1], reverse=True)
            items_to_plot_pos = sorted_pos[:top_n]
            if items_to_plot_pos:
                pos_labels = [f"{item[0]} ({pos_translation_dict.get(item[0], '')})" for item in items_to_plot_pos]
                pos_counts = [item[1] for item in items_to_plot_pos]
                sns.barplot(x=pos_counts, y=pos_labels, palette="mako", hue=pos_labels, ax=ax2, legend=False)
                ax2.set_title(f'Frequency of Top {len(pos_labels)} Part-of-Speech Categories', fontsize=15)
                ax2.set_xlabel('Total Word Count in Corpus', fontsize=12)
                ax2.set_ylabel('Part-of-Speech (POS) Category', fontsize=12)
                if len(pos_counts) > 0 and max(pos_counts, default=0) > 1000:
                    ax2.set_xscale('log')
                    ax2.set_xlabel('Total Word Count in Corpus (Log Scale)', fontsize=12)
                plt.tight_layout()
                try:
                    fig2.savefig(output_dir_path / "pos_major_token_sum_distribution.png", dpi=300)
                    self.logger.info(f"Saved pos_major_token_sum_distribution.png")
                except Exception as e:
                    self.logger.error(f"Failed to save pos_major_token_sum_distribution.png: {e}", exc_info=True)
            plt.close(fig2)
        self.logger.info("Corpus statistics visualizations generation finished.")


if __name__ == "__main__":
    import sys

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # The import does work during runtime
    from wiki_corpus_reader.wiki_corpus_reader import WikiCorpusReader

    # --- Step 1: Set up logging ---
    test_output_base_dir = Path("corpus_processor_test_output")
    log_file_path = test_output_base_dir / "corpus_processor_test.log"
    test_output_base_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file_path, mode='w', encoding='utf-8')]
    )
    test_logger = logging.getLogger("CorpusProcessorTest")
    test_logger.info("=== Starting CorpusProcessor Self-Test Script (with refined logic) ===")

    # --- Step 2: Initialize CorpusProcessor ---
    try:
        corpus_proc = CorpusProcessor()
    except Exception as e_init:
        test_logger.critical(f"CRITICAL: Failed to initialize CorpusProcessor: {e_init}", exc_info=True)
        exit()

    # --- Step 3: Show indices ---
    test_logger.info("\n--- Verifying MeCab Feature Indices for the active dictionary ---")
    test_word = "食べた"
    test_logger.info(f"Analyzing test word '{test_word}' to check feature indices...")
    node = corpus_proc.tagger.parseToNode(test_word)
    node = node.next  # Advance past BOS

    if node and node.surface:
        test_logger.info(f"Analyzing first token: Surface Form = '{node.surface}'")
        test_logger.info(f"Full Feature String: {node.feature}")
        features_list = node.feature.split(',')
        test_logger.info("Feature List with Indices:")
        for i, feature in enumerate(features_list):
            test_logger.info(f"  Index {i}: {feature}")

        configured_lemma_index = corpus_proc.idx_lemma
        if configured_lemma_index < len(features_list):
            extracted_lemma = features_list[configured_lemma_index]
            test_logger.info(
                f"==> Extracted Lemma (using configured index {configured_lemma_index}): '{extracted_lemma}'")
            if extracted_lemma == "食べる":
                test_logger.info("==> SUCCESS: The configured lemma index appears to be correct for '食べる'.")
            else:
                test_logger.warning(
                    f"==> WARNING: The extracted lemma for the first token ('{node.surface}') is '{extracted_lemma}', not '食べる'. Please VERIFY `self.idx_lemma` for your dictionary.")
        else:
            test_logger.error(
                f"==> ERROR: Configured lemma index {configured_lemma_index} is out of bounds for the feature list (length {len(features_list)}).")
    else:
        test_logger.error(f"Could not find any meaningful nodes for the test word '{test_word}'.")

    # --- Step 4: Define corpus path and check existence ---
    WIKI_EXTRACTED_PATH = "/Volumes/T7/Bachelorthesis/jawiki_data/jawiki_extracted"
    if not Path(WIKI_EXTRACTED_PATH).exists():
        test_logger.error(f"Real corpus path not found: {WIKI_EXTRACTED_PATH}. Please update the path and re-run.")
        exit()

    # --- Step 5: Process a sample of the real corpus ---
    NUM_FILES_TO_PROCESS = 2000
    test_logger.info(f"\n--- Preparing to process a sample of {NUM_FILES_TO_PROCESS} real corpus files ---")

    setup_reader = WikiCorpusReader(WIKI_EXTRACTED_PATH)
    all_real_files = setup_reader._find_all_json_files()
    if not all_real_files:
        test_logger.error(f"No JSON files found in {WIKI_EXTRACTED_PATH}. Aborting test.")
        exit()

    files_to_process = all_real_files[:min(len(all_real_files), NUM_FILES_TO_PROCESS)]

    corpus_reader = WikiCorpusReader(WIKI_EXTRACTED_PATH)
    corpus_reader._find_all_json_files = lambda: files_to_process

    test_logger.info(f"Processing sentences from {len(files_to_process)} files...")
    sentence_iterable = corpus_reader.stream_sentences()

    corpus_proc.process_sentences_from_iterable(sentence_iterable)

    # --- Step 6: Verify results ---
    test_logger.info("\n--- Basic Processing Results from Real Data Sample ---")
    test_logger.info(f"Total MeCab nodes yielded (incl. BOS/EOS): {corpus_proc.total_tokens_processed:,}")
    test_logger.info(
        f"Total content tokens (after filtering and combining): {sum(corpus_proc.word_pos_frequencies.values()):,}")
    test_logger.info(f"Unique (base_form, pos_major) types found: {len(corpus_proc.word_pos_frequencies):,}")

    if corpus_proc.word_pos_frequencies:
        test_logger.info("\nTop 10 most frequent (lemma, POS) pairs in the sample data:")
        for item, count in corpus_proc.word_pos_frequencies.most_common(10):
            test_logger.info(f"  ('{item[0]}', '{item[1]}'): {count:,}")

    # --- Step 7: Test saving and loading ---
    test_data_dir = test_output_base_dir / "data"
    test_figures_dir = test_output_base_dir / "figures"

    test_logger.info(f"\n--- Testing Save/Load Functionality (Refined) ---")
    corpus_proc.save_processed_data(output_dir=str(test_data_dir),
                                    details_filename="sample_jp_all_words_details_refined.json",
                                    unknown_log_filename="sample_unknown_words_refined.json")

    loaded_data = CorpusProcessor.load_all_words_details(
        details_file_path_str=str(test_data_dir / "sample_jp_all_words_details_refined.json"))
    if loaded_data:
        test_logger.info(f"SUCCESS: Loaded {len(loaded_data):,} entries from saved JSON.")
        assert len(loaded_data) == len(corpus_proc.word_pos_frequencies), "Loaded data count mismatch!"
    else:
        test_logger.error("FAILURE: Failed to load saved data.")

    # --- Step 8: Test visualization and other utilities ---
    test_logger.info("\n--- Testing Visualization Generation (Refined) ---")
    corpus_proc.generate_corpus_stats_visualizations(output_dir=str(test_figures_dir), top_n=15)

    test_logger.info("\n=== CorpusProcessor Self-Test Script Finished Successfully ===")
