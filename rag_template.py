import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

def load_templates(template_dir=TEMPLATE_DIR):
    template_texts = []
    template_labels = []

    for filename in os.listdir(template_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(template_dir, filename), "r", encoding="utf-8") as f:
                template_texts.append(f.read())
                template_labels.append(filename.replace(".txt", ""))

    return template_texts, template_labels

def build_faiss_index(template_texts, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(template_texts, convert_to_tensor=False)
    # Normalize for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return model, index


def _compare_single_clause(
    clause_text,
    model,
    index,
    template_labels,
    template_texts,
    deviation_threshold
):
    emb = model.encode([clause_text], convert_to_tensor=False)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    D, I = index.search(emb, 1)

    similarity = float(D[0][0])
    template_idx = int(I[0][0])

    return {
        "template_match": template_labels[template_idx],
        "template_similarity": similarity,
        "template_deviation": similarity < deviation_threshold
    }

def compare_with_templates(
    clauses,
    template_dir,
    deviation_threshold=0.75
):
    """
    Input:
        clauses: List[dict] with key 'text'

    Output:
        Same clauses enriched with template comparison
    """

    template_texts, template_labels = load_templates(template_dir)
    model, index = build_faiss_index(template_texts)

    enriched_clauses = []

    for clause in clauses:
        comparison = _compare_single_clause(
            clause_text=clause["text"],
            model=model,
            index=index,
            template_labels=template_labels,
            template_texts=template_texts,
            deviation_threshold=deviation_threshold
        )

        clause.update(comparison)
        enriched_clauses.append(clause)

    return enriched_clauses







