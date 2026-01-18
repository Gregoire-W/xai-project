from core.constants import AUDIO_MODELS_PATH, IMAGE_MODELS_PATH
import json
import tensorflow as tf

class ModelLoaderService:

    def load_model(self, model_name: str, file_type):
        if file_type == "audio":
            base_path = AUDIO_MODELS_PATH
        elif file_type == "image":
            base_path = IMAGE_MODELS_PATH
        model_path = base_path / model_name / "model.keras"
        grad_model_path = base_path / model_name / "grad_model.keras"
        metadata_path = base_path / model_name / "metadata.json"

        if(model_path.exists()):
            model = tf.keras.models.load_model(str(model_path))
            print("Model loaded successfully", flush=True)
        else:
            return { "success": False, "data": {"error": f"no model at path: {model_path}"}, }
        if(grad_model_path.exists()):
            grad_model = tf.keras.models.load_model(str(grad_model_path))
            print("Grad model loaded successfully", flush=True)
        else:
            return { "success": False, "data": {"error": f"no grad model at path: {grad_model_path}"}, }
        if(metadata_path.exists()):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                print(f"metadata: {metadata}", flush=True)
        else:
            return { "success": False, "data": {"error": f"no class names at path: {metadata_path}"}, }
        
        return { "success": True, "data": {"model": model, "grad_model": grad_model, "metadata": metadata}} 
    
model_loader_service = ModelLoaderService()