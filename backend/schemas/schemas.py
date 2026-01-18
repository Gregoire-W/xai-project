from pydantic import BaseModel
from typing import Literal

class AnalyseRequest(BaseModel):
    model: str
    xai_methods: list[str]
    file_type: Literal["image", "audio"]
    file_b64: str