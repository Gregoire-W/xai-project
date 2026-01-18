from fastapi import APIRouter
from core.constants import AUDIO_MODELS_PATH, IMAGE_MODELS_PATH
from services.analysis_service import analysis_service
from schemas.schemas import AnalyseRequest

router = APIRouter()

@router.get("/model_list/", summary="model_list route")
async def model_list() -> dict[str, list]:
    audio = [path.name for path in (AUDIO_MODELS_PATH).iterdir() if path.is_dir()]
    image = [path.name for path in (IMAGE_MODELS_PATH).iterdir() if path.is_dir()]
    return {"audio_model": audio, "image_model": image}

@router.post("/predict/", summary="predict route")
async def predict(request: AnalyseRequest):
    return analysis_service.analyse(
        request.model,
        request.xai_methods,
        request.file_type,
        request.file_b64,
    )