# PURPOSE:
# This module is the core upgrade from the previous system. It analyzes the carrier
# sentence to determine whether it provides a "Closed" or "Open" context for the blank.
#
# HOW IT'S AN UPGRADE:
# The previous system was context-blind, leading to errors where valid answers were
# proposed as distractors (e.g., suggesting "dog" for "My ___ is cute" when the
# target is "cat").
#
# This analyzer uses a powerful pre-trained language model (Japanese BERT) to quantify
# the contextual constraint. By calculating the entropy of the model's predictions
# for the blank space, we can determine how "sure" the model is.
#   - Low Entropy: The model is very certain, meaning the context is "Closed".
#     (e.g., "That ___ meowed." -> blank is almost certainly "cat")
#   - High Entropy: The model is uncertain, meaning the context is "Open".
#     (e.g., "I like my ___." -> blank could be many things)
#
# This classification allows the downstream ranking modules to apply different, more
# appropriate strategies for each context type.
#
# ========================================================================================
import torch
from transformers import BertJapaneseTokenizer, BertForMaskedLM
from scipy.stats import entropy

class ContextAnalyzer:
    """
    Analyzes a carrier sentence to determine if its context is Open or Closed.
    """

    def __init__(self, model_name: str, entropy_threshold: float):
        """
        Initializes the tokenizer and the masked language model from Hugging Face.

        Args:
            model_name (str): The name of the pre-trained Japanese BERT model.
                              Example: 'cl-tohoku/bert-base-japanese-whole-word-masking'
            entropy_threshold (float): The threshold to distinguish between Open and
                                       Closed contexts. This value will need to be
                                       tuned through experimentation.
        """
        print("Initializing ContextAnalyzer...")

        # The tokenizer is responsible for converting text into a format BERT understands,
        # including handling special tokens like [MASK].
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)

        # BertForMaskedLM is the specific BERT architecture designed for predicting
        # masked tokens, which is exactly what we need.
        self.model = BertForMaskedLM.from_pretrained(model_name)

        # Evaluation mode This turns off layers like dropout,
        # which are only used during training, to ensure deterministic outputs.
        self.model.eval()

        # This is a critical hyperparameter. If the calculated entropy is ABOVE this
        # value, we classify the context as "Open". If it's below, it's "Closed".
        self.entropy_threshold = entropy_threshold

        print(f"✅ ContextAnalyzer ready. Entropy threshold set to: {self.entropy_threshold}")


    def analyze_context(self, sentence: str, expected_context_type: str, translated_sentence: str) -> str:
        """
        Calculates the entropy of the word distribution for the masked token
        in a sentence and classifies the context as "Open" or "Closed".

        Args:
            sentence (str): The carrier sentence containing a single '[MASK]' token
                            representing the blank. Example: "私の[MASK]はとても可愛い。"

        Returns:
            str: Either "Open" or "Closed".
        """

        # 1. Tokenize the input sentence
        # The tokenizer converts the string into a list of token IDs and also
        # creates an attention mask to tell the model which tokens to pay attention to.
        inputs = self.tokenizer(sentence, return_tensors='pt')
        token_ids = inputs['input_ids']

        # 2. Find the position of the [MASK] token
        # We need to know which token's output distribution to analyze.
        # The `[MASK]` token has a specific ID in the tokenizer's vocabulary.
        try:
            masked_index = torch.where(token_ids == self.tokenizer.mask_token_id)[1].item()
            print(f"[MASK] token index: {masked_index}")
        except IndexError:
            print(f"Warning: [MASK] token not found in sentence: '{sentence}'")
            return "Error: No [MASK] token"


        # 3. Get model predictions
        # We run the tokenized input through the model. 'with torch.no_grad()' is a
        # performance optimization that tells PyTorch not to calculate gradients,
        # as we are not training the model.
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits

        # 4. Extract and process the probability distribution for the MASK token
        # The model's output `predictions` has shape [batch_size, sequence_length, vocab_size].
        # We isolate the predictions for our specific masked token.
        mask_token_logits = predictions[0, masked_index, :]

        # We apply the softmax function to the raw logits to convert them into a
        # true probability distribution (i.e., all values sum to 1).
        probabilities = torch.nn.functional.softmax(mask_token_logits, dim=-1)

        # 5. Calculate the entropy of the distribution
        # Entropy is a measure of uncertainty or "surprise". A high entropy means
        # the distribution is spread out and uncertain. A low entropy means the
        # distribution is peaked and certain.
        # We use the `entropy` function from scipy for this calculation.
        # We detach the tensor from the computation graph and move it to the CPU to use with NumPy/SciPy.
        calculated_entropy = entropy(probabilities.cpu().numpy())

        # 6. Classify the context based on the threshold
        if calculated_entropy > self.entropy_threshold:
            context_type = "Open"
        else:
            context_type = "Closed"
            
        print(f"Analyzed sentence: '{sentence}'")
        print(f"Translation: {translated_sentence}")
        print(f"  - Calculated Entropy: {calculated_entropy:.4f}")
        print(f"  - Classification: {context_type}")
        print(f"  - Expected Classification:  {expected_context_type}")

        return context_type

if __name__ == '__main__':
    MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    
    # TODO: This threshold is just an example. The optimal value must be found.
    ENTROPY_THRESHOLD = 4.0

    analyzer = ContextAnalyzer(model_name=MODEL_NAME, entropy_threshold=ENTROPY_THRESHOLD)
    print("-" * 50)

    # --- Test Case 1: Expected to be "Closed" (Strong verb constraint) ---
    # The presence of "meowed" (ニャー) strongly implies the blank is "cat".
    closed_context_sentence_1 = "その[MASK]は魚を咥えて、ニャーと鳴いた。"
    analyzer.analyze_context(closed_context_sentence_1, "Closed", "That ___ held a fish in its mouth and meowed.")
    print("-" * 50)

    # --- Test Case 2: Expected to be "Open" (Vague adjective) ---
    # "My ___ is cute" could have many valid answers (dog, hamster, child, etc.).
    open_context_sentence_1 = "私の[MASK]はとても可愛い。"
    analyzer.analyze_context(open_context_sentence_1, "Open", "My ___ is cute.")
    print("-" * 50)

    # --- Test Case 3: Expected to be "Closed" (Strong noun constraint) ---
    # "The capital of Japan is ___." The answer is highly constrained.
    closed_context_sentence_2 = "日本の首都は[MASK]です。"
    analyzer.analyze_context(closed_context_sentence_2, "Closed", "The capital of Japan is ___.")
    print("-" * 50)

    # --- Test Case 4: Expected to be "Open" (General verb) ---
    # "I ___ to the park yesterday." Many actions are possible (went, ran, walked, etc.).
    open_context_sentence_2 = "昨日、公園に[MASK]ました。"
    analyzer.analyze_context(open_context_sentence_2, "Open", "I ___ to the park yesterday.")
    print("-" * 50)
    
    # --- Test Case 5: Expected to be "Closed" (Idiomatic phrase) ---
    # "Seeing a shooting star, he made a ___." This is often "wish".
    closed_context_sentence_3 = "流れ星を見て、彼は[MASK]をかけた。"
    analyzer.analyze_context(closed_context_sentence_3, "Closed", "Seeing a shooting star, he made a ___.")
    print("-" * 50)

    # --- Test Case 6: Expected to be "Open" (Common routine) ---
    open_context_sentence_3 = "朝食に[MASK]を食べます。"
    analyzer.analyze_context(open_context_sentence_3, "Open", "I eat ___ for breakfast.")
    print("-" * 50)

    # --- Test Case 7: Expected to be "Closed" (Scientific fact) ---
    closed_context_sentence_4 = "水はH2Oとしても知られる[MASK]です。"
    analyzer.analyze_context(closed_context_sentence_4, "Closed", "Water is a ___ also known as H2O.")
    print("-" * 50)

    # --- Test Case 8: Expected to be "Open" (Requesting a color) ---
    open_context_sentence_4 = "この壁を[MASK]色に塗りたいです。"
    analyzer.analyze_context(open_context_sentence_4, "Open", "I want to paint this wall a ___ color.")
    print("-" * 50)

    # --- Test Case 9: Expected to be "Closed" (Antonym context) ---
    open_context_sentence_5 = "戦争ではなく[MASK]を。"
    analyzer.analyze_context(open_context_sentence_5, "Open", "Not war, but ___.")
    print("-" * 50)

    # --- Test Case 10: Expected to be "Closed" (Famous Proverb) ---
    closed_context_sentence_5 = "猿も[MASK]から落ちる。"
    analyzer.analyze_context(closed_context_sentence_5, "Closed", "Even monkeys fall from ___.")
    print("-" * 50)