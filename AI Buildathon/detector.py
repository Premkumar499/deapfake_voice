import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Detector(nn.Module):
    
    def __init__(self):
        
        try:
            super().__init__()
            logger.info("Initializing Wav2Vec2 encoder...")
            
            self.encoder = Wav2Vec2Model.from_pretrained(
                "facebook/wav2vec2-base"
            )
            self.classifier = nn.Linear(768, 1)
            
            logger.info("Detector model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Detector: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output logits
            
        Raises:
            RuntimeError: If forward pass fails
        """
        try:
            if x is None:
                raise ValueError("Input tensor is None")
                
            x = self.encoder(x).last_hidden_state
            x = x.mean(dim=1)
            return self.classifier(x)
            
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            raise RuntimeError(f"Model forward pass failed: {e}")