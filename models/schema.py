from pydantic import BaseModel
from typing import List

class EmbeddingCVandPosteInput(BaseModel):
    offre: str
    cvs: list[str]

class RetrieverInput(BaseModel):
    offre_embedding: list[float]
    cvs: list[str]
    cvs_embedding: list[list[float]]
    top_k: int = 5

class GeneratorInput(BaseModel):
    context: str
    question: str

class AnalyseInput(BaseModel):
    offre: str
    cvs: list[str]
    question: str = "Parmis ces CVs, quels sont les meilleurs profils ?"
    top_k: int = 5
