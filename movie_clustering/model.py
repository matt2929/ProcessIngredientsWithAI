from enum import IntEnum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict


class VideoType(IntEnum):
    Movie = 1
    Series = 2


class VideoObj(BaseModel):
    # Pydantic will automatically convert strings to Path objects
    name_raw: str
    path: Path
    synopsis: Optional[str] = None
    video_type: Optional[VideoType] = None
    label: Optional[str] = None
    # Allows the model to interact with arbitrary types if needed
    model_config = ConfigDict(from_attributes=True)
