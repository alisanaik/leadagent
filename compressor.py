import re
import torch
from sentence_transformers import SentenceTransformer, util
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


# Load the MiniLM model once (downloads ~80MB on first run, then cached locally)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None  # Lazy-loaded on first use to speed up imports


def _get_model():
    """Lazy-load the model only when actually needed."""
    global _model
    if _model is None:
        print("[Compressor] Loading MiniLM embedding model...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# Strong leadership titles — generic words like "director" or "head of" cause false positives
STRONG_TITLES = [
    "ceo", "cto", "coo", "cfo", "founder", "founded", "co-founder",
    "cofounder", "chief executive", "chief technology", "chief operating",
    "president", "managing director",
]

# Match proper names: two capitalized words, but exclude common false positives
NAME_PATTERN = re.compile(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b")
FALSE_POSITIVES = {
    "United States", "New York", "San Francisco", "Los Angeles",
    "Postman Inc", "Supabase Inc", "Vapi Inc", "Silicon Valley",
    "Postman Company", "Product Manager", "Software Engineer",
}

# Never force more than this many chunks — prevents defeating compression entirely
MAX_FORCED = 15


def compress_text(text: str, query: str, top_k: int = 10, chunk_size: int = 400) -> str:
    """
    Splits text into chunks, embeds them, and returns the top_k most relevant chunks
    as a single concatenated string.

    Also force-includes a LIMITED number of chunks that look like leadership bios,
    so founder names are never missed even if semantic similarity ranks them low.
    """
    # 1. Split the text into chunks of ~chunk_size characters.
    # We split by paragraphs first to preserve semantic boundaries, then merge short ones.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += "\n\n" + para
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para
    if current_chunk:
        chunks.append(current_chunk.strip())

    # If the text is already small, skip compression entirely
    if len(chunks) <= top_k:
        return text

    # 2. Embed all chunks and the query using the local MiniLM model
    model = _get_model()
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True, show_progress_bar=False)
    query_embedding = model.encode(query, convert_to_tensor=True, show_progress_bar=False)

    # 3. Compute cosine similarity between the query and each chunk
    similarities = util.cos_sim(query_embedding, chunk_embeddings)[0]

    # 4. Pick the indices of the top_k most similar chunks
    top_indices = torch.topk(similarities, k=min(top_k, len(chunks))).indices.tolist()

    # 5. Sort to preserve reading order (better for the LLM)
    top_indices.sort()

    # SAFETY NET: Force-include leadership bio chunks, but be selective.
    # A chunk only qualifies if it has BOTH a strong title AND a real name.
    forced_indices = set()
    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        has_title = any(t in chunk_lower for t in STRONG_TITLES)
        if not has_title:
            continue
        names = NAME_PATTERN.findall(chunk)
        has_real_name = any(n not in FALSE_POSITIVES for n in names)
        if has_real_name:
            forced_indices.add(i)

    # Force-include any chunk containing a LinkedIn profile URL.
    # This catches bios where the URL is separated from the name/title across chunks.
    for i, chunk in enumerate(chunks):
        if "linkedin.com/in" in chunk.lower():
            forced_indices.add(i)
    
    # Cap: never force more than MAX_FORCED chunks, even if many match.
    if len(forced_indices) > MAX_FORCED:
        forced_indices = set(sorted(forced_indices)[:MAX_FORCED])

    # 6. Merge with top_k results, dedupe, and sort to preserve reading order
    all_indices = sorted(set(top_indices) | forced_indices)
    compressed = "\n\n".join(chunks[i] for i in all_indices)

    print(
        f"[Compressor] Reduced text from {len(text)} chars to {len(compressed)} chars "
        f"(kept {len(all_indices)}/{len(chunks)} chunks, {len(forced_indices)} forced for leadership)."
    )

    return compressed