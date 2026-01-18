from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.router import api_router


app = FastAPI(title="XAI backend")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifiez les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
