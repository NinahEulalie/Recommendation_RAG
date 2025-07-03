from fastapi import FastAPI
from routes.rag_routes import router

app = FastAPI(
    title="Système Intelligent RAG"
)

app.include_router(router)