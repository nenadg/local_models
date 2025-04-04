from transformers import LogitsProcessor, LogitsProcessorList
import torch

class TokenProbabilityCaptureProcessor(LogitsProcessor):
    """
    A LogitsProcessor that captures token probabilities during generation.

    This processor captures the probabilities and entropy for each token
    to provide accurate confidence metrics for the generated text.
    """
    def __init__(self, confidence_metrics):
        """
        Initialize the processor with a reference to the confidence metrics object.

        Args:
            confidence_metrics: ConfidenceMetrics instance to store token scores
        """
        self.confidence_metrics = confidence_metrics

    def __call__(self, input_ids, scores):
        """
        Process logits during generation and capture metrics.

        Args:
            input_ids: Current sequence of input IDs
            scores: Logits for the next token prediction

        Returns:
            The original scores (unchanged)
        """
        # For each item in the batch
        for i in range(scores.shape[0]):
            # Get most likely token id
            token_id = torch.argmax(scores[i]).item()

            # Record metrics for this token
            self.confidence_metrics.add_token_score(scores[i], token_id)

        # Return scores unchanged (we're just observing)
        return scores