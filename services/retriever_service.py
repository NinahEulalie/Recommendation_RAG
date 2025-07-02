import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def retrieve_similar(offre_emb, cvs_emb, cvs, top_k=3):
    similarities = cosine_similarity([offre_emb], cvs_emb)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [
        {"cv": cvs[i], "score": float(similarities[i])}
        for i in top_indices
    ]