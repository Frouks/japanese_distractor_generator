import torch
import logging
from transformers import AutoTokenizer, AutoModelForMaskedLM

class ContextAnalyzer:
    """
    Analyzes the context of a carrier sentence to determine if it is "Open" or "Closed".
    This is the core component of the Context-Aware Distractor Generation System (CADGS).
    """

    def __init__(self, model_name='cl-tohoku/bert-base-japanese-whole-word-masking'):
        """
        Initializes the ContextAnalyzer by loading a pre-trained BERT model.

        WHAT:
        - We load a pre-trained Japanese BERT model and its corresponding tokenizer from Hugging Face.
        - We set up a logger for this class.

        WHY:
        - Using a powerful pre-trained model like BERT allows us to leverage its deep understanding of
          Japanese grammar and semantics without having to train a model from scratch.
        - The chosen model is a standard and effective base model for Japanese NLP tasks.

        ADVANTAGE:
        - This approach is highly robust. The model's predictions for a masked token are a direct
          and powerful signal of how constrained a sentence's context is.
        - By encapsulating this logic in a class, we can easily reuse it throughout the system.
        """
        self.logger = logging.getLogger('ContextAnalyzer')
        self.logger.info(f"Initializing ContextAnalyzer with model: {model_name}")
    
        try:
            # Load the tokenizer and model using the recommended AutoClass.
            # This automatically handles the specific requirements of the model.
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)
            self.model.eval()  # Set the model to evaluation mode
        
            # Get the device (GPU if available, otherwise CPU)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        
            self.mask_token = self.tokenizer.mask_token
            self.mask_token_id = self.tokenizer.mask_token_id
        
            self.logger.info(f"✅ ContextAnalyzer initialized successfully on device: {self.device}")

        except Exception as e:
            self.logger.error(f"Failed to load model or tokenizer: {e}", exc_info=True)
            # Add a helpful hint for the user if they still face issues
            self.logger.error("Hint: Ensure 'pip install unidic-lite' is run in your environment.")
            raise

    def analyze_context(self, sentence_with_blank, placeholder="___", top_k=5, entropy_threshold=4.5):
        """
        Analyzes a sentence with a blank to determine its context type and returns analysis results.
        (No changes to this method's logic)
        """
        if placeholder not in sentence_with_blank:
            self.logger.error(f"Placeholder '{placeholder}' not found in the sentence.")
            return None

        # 1. Prepare the input sentence for BERT
        masked_sentence = sentence_with_blank.replace(placeholder, self.mask_token, 1)
        self.logger.debug(f"Analyzing masked sentence: {masked_sentence}")

        inputs = self.tokenizer(masked_sentence, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        try:
            mask_token_index = torch.where(input_ids == self.mask_token_id)[1][0]
        except IndexError:
            self.logger.error(f"Could not find [MASK] token in the tokenized input: {self.tokenizer.convert_ids_to_tokens(input_ids[0])}")
            return None

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits

        masked_token_logits = predictions[0, mask_token_index, :]
        probabilities = torch.nn.functional.softmax(masked_token_logits, dim=-1)

        entropy = torch.distributions.Categorical(probs=probabilities).entropy().item()
        context_type = "Open" if entropy > entropy_threshold else "Closed"

        top_k_probs, top_k_indices = torch.topk(probabilities, top_k)
        top_k_tokens = self.tokenizer.convert_ids_to_tokens(top_k_indices)
        top_predictions = list(zip(top_k_tokens, top_k_probs.cpu().numpy()))

        self.logger.info(f"Context analysis complete. Type: {context_type}, Entropy: {entropy:.4f}")

        return {
            "context_type": context_type,
            "entropy": entropy,
            "top_predictions": top_predictions
        }

# ==============================================================================
#  TESTING BLOCK
# ==============================================================================
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    test_logger = logging.getLogger("ContextAnalyzerTest")
    test_logger.info("=== Running ContextAnalyzer Self-Test Script (Final Version) ===")

    try:
        analyzer = ContextAnalyzer()
    except Exception as e:
        test_logger.critical("Failed to initialize the analyzer. Please check your internet connection and dependencies.", exc_info=True)
        exit()

    open_context_sentence = "私の___はとても可愛い。"
    closed_context_sentence_1 = "その___は魚を咥えて、ニャーと鳴いた。"
    closed_context_sentence_2 = "彼は100メートルを11秒で___ことができる。"

    test_sentences = {
        "Open Example": open_context_sentence,
        "Closed Example (Cat)": closed_context_sentence_1,
        "Closed Example (Run)": closed_context_sentence_2,
    }
    
    for name, sentence in test_sentences.items():
        print("\n" + "="*50)
        test_logger.info(f"Testing: {name} - '{sentence}'")
        analysis_result = analyzer.analyze_context(sentence, entropy_threshold=4.5)
        
        if analysis_result:
            print(f"  Context Type Detected: {analysis_result['context_type']}")
            print(f"  Calculated Entropy: {analysis_result['entropy']:.4f}")
            print(f"  Top 5 BERT Predictions:")
            for token, prob in analysis_result['top_predictions']:
                print(f"    - {token:<10} (Probability: {prob:.4f})")
    
    test_logger.info("\n=== ContextAnalyzer Self-Test Finished Successfully ===")