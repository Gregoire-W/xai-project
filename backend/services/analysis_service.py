from typing import Literal
from .audio_xai_service import audio_xai_service
from .image_xai_service import image_xai_service
from .utils import audio_to_img, b64_to_img
from .model_loader_service import model_loader_service
from .classification_service import classification_service
from io import BytesIO
import base64

class AnalysisService:

    def analyse(
        self,
        model: str,
        xai_methods: list[Literal["LIME", "Grad-CAM", "SHAP"]],
        file_type: Literal["image", "audio"],
        file_b64: str
    ):

        response = model_loader_service.load_model(model, file_type)
        if not response["success"]:
            return response
        else:
            model, grad_model, metadata = (
                response["data"]["model"],
                response["data"]["grad_model"],
                response["data"]["metadata"]
            )

        if file_type == "audio":
            img = audio_to_img(file_b64)
            img_resized = img.resize((224, 224))
            prediction_labels = classification_service.predict_audio(model, metadata["class_names"], img_resized)
        elif file_type == "image":
            img = b64_to_img(file_b64)
            img_resized = img.resize((224, 224))
            prediction_labels = classification_service.predict_image(model, metadata["class_names"], img_resized)


        buf = BytesIO()
        img_resized.save(buf, format="PNG")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")

        xai_results = {}
        for xai in xai_methods:
            if file_type == "audio":
                result = audio_xai_service.explain(xai, model, grad_model, metadata, img_resized)
            elif file_type == "image":
                result = image_xai_service.explain(xai, model, grad_model, metadata, img_resized)
            if result["success"]:
                xai_results[xai] = result["image"]
            else:
                print(f"XAI method {xai} failed: {result.get('error', 'Unknown error')}", flush=True)
        print("Success, returning answer", flush=True)
        return {
            "success": True,
            "data": {
                "prediction": prediction_labels,
                "image": img_b64,
                "xai_results": xai_results,
            },
        }
    
analysis_service = AnalysisService()