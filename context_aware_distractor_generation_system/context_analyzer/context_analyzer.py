# ========================================================================================
# PURPOSE:
# This module is the core upgrade from the previous system. It analyzes the carrier
# sentence to determine whether it provides a "Closed" or "Open" context for the blank.
#
# HOW IT'S AN UPGRADE:
# The previous system was context-blind, leading to errors where valid answers were
# proposed as distractors (e.g., suggesting "dog" for "My ___ is cute" when the
# target is "cat").
#
# This analyzer uses a pre-trained language model (Japanese BERT) to quantify
# the contextual constraint. It now includes TWO methods for this analysis:
#
#   1. Entropy Method: Calculates the overall "uncertainty" of the entire prediction
#      distribution. A low score means the model is very certain (Closed Context).
#
#   2. Top-k Probability Mass: Calculates the combined probability of the top 'k'
#      most likely predictions. If this combined probability is very high, it means
#      the model is confident in a small set of answers (Closed Context), even if
#      the overall entropy is high due to a long tail of improbable words.
#
# ========================================================================================
import torch
from transformers import BertJapaneseTokenizer, BertForMaskedLM
from scipy.stats import entropy

class ContextAnalyzer:
    """
    Analyzes a carrier sentence to determine if its context is Open or Closed
    using two different methods: Entropy and Top-k Probability Mass.
    """

    def __init__(self, model_name: str, entropy_threshold: float, top_k_value: int, top_k_threshold: float):
        """
        Initializes the tokenizer and the masked language model from Hugging Face.

        Args:
            model_name (str): The name of the pre-trained Japanese BERT model.
            entropy_threshold (float): The threshold for the entropy method.
            top_k_value (int): The number of top predictions to consider (e.g., 5).
            top_k_threshold (float): The probability mass threshold for the top-k method.
        """
        print("Initializing ContextAnalyzer...")

        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name)
        # Set the model to evaluation mode to ensure deterministic outputs.
        self.model.eval()

        # --- Thresholds for both analysis methods ---
        self.entropy_threshold = entropy_threshold
        self.top_k_value = top_k_value
        self.top_k_threshold = top_k_threshold

        print(f"✅ ContextAnalyzer ready.")
        print(f"  - Entropy Threshold set to: {self.entropy_threshold}")
        print(f"  - Top-K Value set to: {self.top_k_value}")
        print(f"  - Top-K Probability Threshold set to: {self.top_k_threshold}")

    def _get_probabilities(self, sentence: str) -> tuple[torch.Tensor, int] | tuple[None, None]:
        """A helper function to get the probability distribution for a sentence."""
        inputs = self.tokenizer(sentence, return_tensors='pt')
        token_ids = inputs['input_ids']

        try:
            masked_index = torch.where(token_ids == self.tokenizer.mask_token_id)[1].item()
        except IndexError:
            print(f"Warning: [MASK] token not found in sentence: '{sentence}'")
            return None, None

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits

        mask_token_logits = predictions[0, masked_index, :]
        probabilities = torch.nn.functional.softmax(mask_token_logits, dim=-1)
        return probabilities, masked_index

    def analyze_context_by_entropy(self, sentence: str, expected_context: str, translated_sentence: str):
        """
        Classifies the context based on the entropy of the entire probability distribution.
        """
        probabilities, _ = self._get_probabilities(sentence)
        if probabilities is None:
            return

        calculated_entropy = entropy(probabilities.cpu().numpy())
        predicted_context = "Open" if calculated_entropy > self.entropy_threshold else "Closed"

        print(f"--- Method 1: Entropy Analysis ---")
        print(f"'{translated_sentence}'")
        print(f"  - Calculated Entropy: {calculated_entropy:.4f} (Threshold: > {self.entropy_threshold} for Open)")
        print(f"  - Predicted: {predicted_context} (Expected: {expected_context})")
        print(f"  - Correct? {'✅' if predicted_context == expected_context else '❌'}")


    def analyze_context_by_top_k(self, sentence: str, expected_context: str, translated_sentence: str):
        """
        Classifies the context based on the cumulative probability of the top K predictions.
        """
        probabilities, _ = self._get_probabilities(sentence)
        if probabilities is None:
            return

        # Take the top 'k' probabilities and sum them up.
        top_k_probs, _ = torch.topk(probabilities, self.top_k_value)
        cumulative_prob = torch.sum(top_k_probs).item()
        
        # If the top words capture most of the probability mass, the context is Closed.
        predicted_context = "Closed" if cumulative_prob > self.top_k_threshold else "Open"

        print(f"--- Method 2: Top-{self.top_k_value} Probability Mass Analysis ---")
        print(f"'{translated_sentence}'")
        print(f"  - Cumulative Probability: {cumulative_prob:.4f} (Threshold: > {self.top_k_threshold} for Closed)")
        print(f"  - Predicted: {predicted_context} (Expected: {expected_context})")
        print(f"  - Correct? {'✅' if predicted_context == expected_context else '❌'}")

if __name__ == '__main__':
    MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    
    # --- HYPERPARAMETERS FOR ANALYSIS ---
    # TODO: We need to fine tune these values
    ENTROPY_THRESHOLD = 4.5  
    TOP_K_VALUE = 5          
    TOP_K_THRESHOLD = 0.50   # If the top 5 words have > 50% probability, we call it "Closed"

    # Initialize the analyzer with parameters for both methods.
    analyzer = ContextAnalyzer(
        model_name=MODEL_NAME,
        entropy_threshold=ENTROPY_THRESHOLD,
        top_k_value=TOP_K_VALUE,
        top_k_threshold=TOP_K_THRESHOLD
    )
    
    test_cases = [
        ("私の[MASK]はとても可愛い。", "Open", "My ___ is cute."),
        ("その[MASK]は魚を咥えて、ニャーと鳴いた。", "Closed", "That ___ held a fish in its mouth and meowed."),
        ("日本の首都は[MASK]です。", "Closed", "The capital of Japan is ___."),
        ("昨日、公園に[MASK]ました。", "Open", "I ___ to the park yesterday."),
        ("流れ星を見て、彼は[MASK]をかけた。", "Closed", "Seeing a shooting star, he made a ___."),
        ("朝食に[MASK]を食べます。", "Open", "I eat ___ for breakfast."),
        ("水はH2Oとしても知られる[MASK]です。", "Closed", "Water is a ___ also known as H2O."),
        ("この壁を[MASK]色に塗りたいです。", "Open", "I want to paint this wall a ___ color."),
        ("戦争ではなく[MASK]を。", "Open", "Not war, but ___."), # Open or closed?
        ("猿も[MASK]から落ちる。", "Closed", "Even monkeys fall from ___."),
        ("彼はその知らせを聞いて[MASK]なった。", "Closed", "Hearing that news, he became ___."),
        ("私は[MASK]が好きです。", "Open", "I like ___."),
        ("光合成は、[MASK]が光エネルギーを使って有機物を合成するプロセスです。", "Closed", "Photosynthesis is the process where ___ use light energy to synthesize organic matter."),
        ("私は[MASK]を読みます。", "Open", "I read a(n) ___."),
        ("空港で[MASK]が離陸するのを見た。", "Closed", "I saw the ___ take off at the airport."),
        ("最も大切なのは[MASK]です。", "Open", "The most important thing is ___."),
        ("ハサミで紙を[MASK]。", "Closed", "I ___ the paper with scissors."),
        ("私の趣味は[MASK]です。", "Open", "My hobby is ___."),
        ("毎朝、私は[MASK]を磨きます。", "Closed", "Every morning, I brush my ___."),
        ("何か[MASK]を飲みませんか？", "Open", "Won't you drink some kind of ___?"),
    ]

    for sentence, expected, translation in test_cases:
        print("-" * 60)
        print(f"Analyzing Sentence: {sentence}")
        analyzer.analyze_context_by_entropy(sentence, expected, translation)
        print() 
        analyzer.analyze_context_by_top_k(sentence, expected, translation)
        
    print("-" * 60)