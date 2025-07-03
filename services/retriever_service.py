import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Recherche contextuelle (Retriever)
# def retrieve_similar_texts(offre_emb, cvs_emb, cvs, top_k=3):
#     similarities = cosine_similarity([offre_emb], cvs_emb)[0]
#     top_indices = np.argsort(similarities)[-top_k:][::-1]
#     return [
#         {"cv": cvs[i], "score": float(similarities[i])}
#         for i in top_indices
#     ]
def retrieve_similar_texts(query_embed: np.ndarray, doc_embeddings: list[list[float]], documents: list[str], top_k=3):
    doc_embeddings = np.array(doc_embeddings)
    similarities = cosine_similarity([query_embed], doc_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [
        {"cv": documents[i], "score": float(similarities[i])}
        for i in top_indices
    ]
