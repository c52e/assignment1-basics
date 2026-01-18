import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")
from .train_bpe import bpe_tokenizer
from .tokenizer import Tokenizer
from .transformer import Linear, Embedding, RMSNorm, SwiGLU, RotaryPositionalEmbedding, softmax, scaled_dot_product_attention, MultiheadSelfAttention, TransformerBlock, TransformerLM
