from sentence_transformers import SentenceTransformer, util
import numpy as np

class ClauseClassifier:
    def __init__(self):
        # Strong general-purpose embedding model
        self.model = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2"
        )

        # Canonical clause labels
        self.labels = [
            "Liability",
            "Termination",
            "Confidentiality",
            "Governing Law",
            "Payment",
            "Indemnity",
            "Intellectual Property"
        ]

        self.label_embeddings = self.model.encode(
            self.labels,
            convert_to_tensor=True
        )

    def classify_clause(self, clause_text: str):
        clause_embedding = self.model.encode(
            clause_text,
            convert_to_tensor=True
        )

        similarities = util.cos_sim(
            clause_embedding,
            self.label_embeddings
        )[0]

        best_idx = int(np.argmax(similarities))
        confidence = float(similarities[best_idx])
        if confidence < 0.40:
            predicted_label = "Other"
        else:
            predicted_label = self.labels[best_idx]

        return {
            "predicted_label": predicted_label,
            "confidence": round(confidence, 3)
        }

    def classify_clauses(self, clauses):
        results = []
        for clause in clauses:
            result = self.classify_clause(clause["text"])
            clause["category"] = result["predicted_label"]
            clause["classification_confidence"] = result["confidence"]
            results.append(clause)
        return results
