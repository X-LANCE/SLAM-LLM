import torch
import os
import logging

logger = logging.getLogger(__name__)

def partial_freeze_weights(model, original_vocabsize, total_vocabsize):
    if int(os.environ.get("RANK", "0")) == 0:
        logger.info("Only training partial embedding layer")

    trainable_range = (original_vocabsize, total_vocabsize)

    # Define a hook to zero out the gradient for weights outside the trainable range during the backward pass
    def zero_out_gradient(grad):
        grad[:trainable_range[0], :] = 0
        grad[trainable_range[1] + 1:, :] = 0
        return grad

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Assuming the output layer is `lm_head`
    for param in model.llm.lm_head.parameters():
        # Compute the standard deviation for He initialization
        std_dev = (2.0 / param.size(1)) ** 0.5

        # Initialize the specific rows with He initialization
        param[original_vocabsize:total_vocabsize] = (
            torch.randn((trainable_range[1] - trainable_range[0], param.size(1))) * std_dev
        )
        param.requires_grad = True

        # Register the hook on the weight tensor
        param.register_hook(zero_out_gradient)

def train_embedding_layer_only(model):
    if int(os.environ.get("RANK", "0")) == 0:
        logger.info("Only training embedding layer")

    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.llm.lm_head.parameters():
        param.requires_grad = True