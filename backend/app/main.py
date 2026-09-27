from fastapi import FastAPI

from backend.app.api.routes.chat import router as chat_router
from backend.app.api.routes.documents import router as documents_router
from backend.app.api.routes.health import router as health_router


app = FastAPI(title="Document Chat API", version="0.1.0")
app.include_router(health_router)
app.include_router(documents_router)
app.include_router(chat_router)
