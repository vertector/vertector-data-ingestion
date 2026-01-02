"""Utility functions for model detection and configuration."""

from loguru import logger


def requires_left_padding(model_name: str) -> bool:
    """
    Detect if a model requires left padding based on the name.

    Some models (Qwen, Nemotron) require left padding for proper tokenization.

    Args:
        model_name: Hugging Face model name or path

    Returns:
        True if the model requires left padding

    Examples:
        >>> requires_left_padding("Qwen/Qwen3-Embedding-4B")
        True
        >>> requires_left_padding("nvidia/llama-embed-nemotron-8b")
        True
        >>> requires_left_padding("sentence-transformers/all-MiniLM-L6-v2")
        False
    """
    model_name_lower = model_name.lower()

    # Models that require left padding
    left_padding_indicators = [
        "qwen3",  # Qwen3 family only
        "nemotron",  # NVIDIA Nemotron family
        "llama-embed",  # NVIDIA LLaMA embedding models
    ]

    return any(indicator in model_name_lower for indicator in left_padding_indicators)


def get_padding_side(model_name: str) -> str:
    """
    Get the appropriate padding side for a tokenizer model.

    Args:
        model_name: Hugging Face model name or path

    Returns:
        "left" for models that require it, "right" for others

    Examples:
        >>> get_padding_side("Qwen/Qwen3-Embedding-4B")
        'left'
        >>> get_padding_side("nvidia/llama-embed-nemotron-8b")
        'left'
        >>> get_padding_side("sentence-transformers/all-MiniLM-L6-v2")
        'right'
    """
    return "left" if requires_left_padding(model_name) else "right"


def get_embedding_dimension(model_name: str) -> int | None:
    """
    Extract embedding dimension from a model.

    Tries multiple methods to determine the embedding dimension:
    1. Check known model dimensions
    2. Try to load model config from Hugging Face
    3. Return None if unknown

    Args:
        model_name: Hugging Face model name or path

    Returns:
        Embedding dimension if found, None otherwise

    Examples:
        >>> get_embedding_dimension("Qwen/Qwen3-Embedding-4B")
        2560
        >>> get_embedding_dimension("nvidia/llama-embed-nemotron-8b")
        4096
    """
    # Known dimensions for common models (from MTEB leaderboard)
    known_dimensions = {
        # Qwen3 models
        "qwen/qwen3-embedding-0.6b": 1024,
        "qwen/qwen3-embedding-4b": 2560,
        "qwen/qwen3-embedding-8b": 4096,
        # Nemotron models
        "nvidia/llama-embed-nemotron-8b": 4096,
        # Multilingual E5
        "intfloat/multilingual-e5-large-instruct": 1024,
        # SentenceTransformers models
        "sentence-transformers/all-minilm-l6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "sentence-transformers/all-minilm-l12-v2": 384,
    }

    model_key = model_name.lower()
    if model_key in known_dimensions:
        return known_dimensions[model_key]

    # Try to auto-detect from model config
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name)

        # Different models store dimension in different places
        if hasattr(config, "hidden_size"):
            dim = config.hidden_size
            logger.info(f"Auto-detected embedding dimension for {model_name}: {dim}")
            return dim
        elif hasattr(config, "dim"):
            dim = config.dim
            logger.info(f"Auto-detected embedding dimension for {model_name}: {dim}")
            return dim
        elif hasattr(config, "embedding_size"):
            dim = config.embedding_size
            logger.info(f"Auto-detected embedding dimension for {model_name}: {dim}")
            return dim
        elif hasattr(config, "d_model"):
            dim = config.d_model
            logger.info(f"Auto-detected embedding dimension for {model_name}: {dim}")
            return dim

    except Exception as e:
        logger.debug(f"Could not auto-detect embedding dimension for {model_name}: {e}")

    logger.warning(
        f"Could not determine embedding dimension for {model_name}. "
        "Please specify it manually in VectorStoreConfig."
    )
    return None
