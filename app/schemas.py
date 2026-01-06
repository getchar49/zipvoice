from pydantic import BaseModel, Field
from typing import Optional

class TTSJobCreate(BaseModel):
    text: str = Field("Xin chào các bạn", min_length=1)
    voice_id: str = Field("zipvoice1", description="ID from /v1/voices")
    speed: float = 1.0
    remove_long_sil: bool = False
    num_step: Optional[int] = 16
    guidance_scale: Optional[float] = 1.0
    audio_type: Optional[str] = Field("mp3", description="Audio format: 'wav' or 'mp3'")

class JobCreateResponse(BaseModel):
    job_id: str
    status_url: str
    audio_url: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    file_path: Optional[str] = None
    audio_url: Optional[str] = None
    error: Optional[str] = None