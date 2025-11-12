from sentence_transformers import SentenceTransformer, util
import torch
sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
def pass_filter(src,tgt,th=0.85):
    emb=sbert.encode([src,tgt],convert_to_tensor=True,normalize_embeddings=True)
    sim=float(util.cos_sim(emb[0],emb[1]))
    return sim>=th,{"sim":sim}
