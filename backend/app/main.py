from fastapi import FastAPI

from backend.app.api.routes.health import router as health_router


app = FastAPI(title="Document Chat API", version="0.1.0")
app.include_router(health_router)
