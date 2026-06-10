def split_prefill_length(total_tokens: int, chunk_size: int | None) -> list[int]:
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if total_tokens <= 0:
        return []
    if chunk_size is None or total_tokens <= chunk_size:
        return [total_tokens]
    chunks: list[int] = []
    remaining = total_tokens
    while remaining > 0:
        chunk = min(chunk_size, remaining)
        chunks.append(chunk)
        remaining -= chunk
    return chunks
