import os
import json
import numpy as np
import faiss
from pathlib import Path
from segmenter import Clause

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, "vector_store", "faiss_index", "clauses.index")
METADATA_PATH = os.path.join(PROJECT_ROOT, "vector_store", "faiss_index", "metadata.json")


# --- Embedding (lazy loaded) ---

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedder

def embed(texts: list[str]) -> np.ndarray:
    model = get_embedder()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


# --- Index loading ---

def load_index():
    """Load FAISS index built by build_cuad_index.py. Must be run first."""
    if not Path(FAISS_INDEX_PATH).exists() or not Path(METADATA_PATH).exists():
        raise FileNotFoundError(
            "FAISS index not found. Please run build_cuad_index.py first."
        )
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    return index, metadata


# --- MMR Search ---

def mmr_search(query_text: str, index, metadata: list, top_k: int = 3, fetch_k: int = 20, lambda_mult: float = 0.5) -> list[dict]:
    """
    MMR retrieval — balances similarity to query vs diversity among results.
    fetch_k    : candidate pool size fetched from FAISS before MMR re-ranking
    lambda_mult: 0.0 = max diversity, 1.0 = max similarity (0.5 is balanced)
    """
    query_vec = embed([query_text])

    # Step 1: fetch candidate pool from FAISS
    scores, indices = index.search(query_vec, min(fetch_k, index.ntotal))

    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        candidates.append({
            "idx": int(idx),
            "clause_type": metadata[idx]["clause_type"],
            "reference_text": metadata[idx]["text"],
            "similarity": float(score)
        })

    if not candidates:
        return []

    # Step 2: get embeddings of all candidates
    candidate_embeddings = np.array([index.reconstruct(c["idx"]) for c in candidates])

    # Step 3: MMR selection loop
    selected = []
    remaining = list(range(len(candidates)))

    while len(selected) < top_k and remaining:
        best_score = -float("inf")
        best_pos = None

        for pos in remaining:
            sim_to_query = candidates[pos]["similarity"]

            if selected:
                selected_embeddings = candidate_embeddings[[s for s in selected]]
                redundancy = float(np.max(candidate_embeddings[pos] @ selected_embeddings.T))
            else:
                redundancy = 0.0

            mmr_score = lambda_mult * sim_to_query - (1 - lambda_mult) * redundancy

            if mmr_score > best_score:
                best_score = mmr_score
                best_pos = pos

        selected.append(best_pos)
        remaining.remove(best_pos)

    return [candidates[i] for i in selected]


# --- Similarity thresholds ---

SIMILARITY_GOOD   = 0.75   # reliable match — deviation score is trustworthy
SIMILARITY_WEAK   = 0.50   # approximate match — use with caution
# below SIMILARITY_WEAK = no meaningful match — clause is highly unusual


def get_match_quality(similarity: float) -> str:
    """Returns match quality label based on similarity score."""
    if similarity >= SIMILARITY_GOOD:
        return "good"
    elif similarity >= SIMILARITY_WEAK:
        return "weak"
    else:
        return "none"


def compute_deviation_score(similarity: float) -> float:
    """Convert similarity (0-1) to deviation score (0-1). Higher = more deviant."""
    return round(1.0 - similarity, 4)


# --- Main entry point ---

def run_rag(clauses: list[Clause]) -> list[Clause]:
    """Attach deviation_score, match_quality and top matching reference to each clause."""
    index, metadata = load_index()

    for clause in clauses:
        if not clause.text.strip():
            continue

        results = mmr_search(clause.text, index, metadata, top_k=3, fetch_k=20, lambda_mult=0.5)

        if results:
            top_match = results[0]
            similarity = top_match["similarity"]
            clause.deviation_score = compute_deviation_score(similarity)
            clause.top_reference = top_match["reference_text"]
            clause.reference_similarity = similarity
            clause.match_quality = get_match_quality(similarity)
        else:
            clause.deviation_score = 1.0
            clause.top_reference = None
            clause.reference_similarity = 0.0
            clause.match_quality = "none"

    return clauses