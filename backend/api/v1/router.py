from fastapi import APIRouter
from api.v1.endpoints import classification_models

router = APIRouter()
router.include_router(classification_models.router, prefix="/classification_models")