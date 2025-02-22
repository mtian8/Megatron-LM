import importlib.metadata
# __version__ = importlib.metadata.version(__package__ or __name__)

from customized_dkernel.interface import SparseAttention, LocalStrideSparseAttention
from customized_dkernel.utils import get_sparse_attn_mask

__all__ = [
    "SparseAttention",
    "LocalStrideSparseAttention",
    "get_sparse_attn_mask",
    ]
