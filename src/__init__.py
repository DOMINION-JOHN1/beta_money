from fastapi import FastAPI
from pydantic import BaseModel

from src.ai_app import RAGSystem


app = FastAPI()
rag_service = RAGSystem()


class UserQuery(BaseModel):
    query: str


@app.get("/")
async def read_root():
    return {"message": "Hello, Welcome to our RAG system"}


@app.post("/chat/text")
async def text_chat(user_query: UserQuery) -> dict:
    query_context = rag_service.retrieve_documents(user_query.query)
    ai_response = rag_service.generate_response(user_query.query, query_context)
    return {"response": ai_response}


@app.get("/chat/audio")
async def audio_chat() -> dict:
    pass


@app.get("/chat/video")
async def video_chat() -> dict:
    pass
