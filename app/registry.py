import os
import json # Added import
from dataclasses import dataclass
from typing import Dict, List, Optional # Added Optional

from .settings import VOICES_DIR

@dataclass(frozen=True)
class Voice:
    voice_id: str
    prompt_wav: str
    prompt_text: str
    # New fields for metadata
    name: str
    gender: str
    region: str
    language: str

class VoiceRegistry:
    def __init__(self, root: str = VOICES_DIR):
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        self._voices: Dict[str, Voice] = {}
        self.refresh()

    def refresh(self) -> None:
        self._voices.clear()
        for name in sorted(os.listdir(self.root)):
            d = os.path.join(self.root, name)
            if not os.path.isdir(d): 
                continue

            wav = os.path.join(d, "prompt.wav")
            txt_path = os.path.join(d, "prompt.txt")
            meta_path = os.path.join(d, "metadata.json") # New check

            if os.path.isfile(wav) and os.path.isfile(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    txt = f.read().strip()
                
                # --- Default Strategy (Convention) ---
                # Guess details from the folder name (e.g., "nu_bac_1")
                display_name = name
                gender = "unknown"
                region = "unknown"
                language = "vi"

                # Simple heuristic if you want to keep the "smart" guessing here
                lower_name = name.lower()
                if "nu" in lower_name or "female" in lower_name:
                    gender = "female"
                elif "nam" in lower_name or "male" in lower_name:
                    gender = "male"
                
                if "bac" in lower_name: region = "North"
                elif "trung" in lower_name: region = "Central"
                elif "nam" in lower_name: region = "South"

                # --- Override Strategy (Configuration) ---
                # If metadata.json exists, use it to overwrite defaults
                if os.path.isfile(meta_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                            display_name = meta.get("name", display_name)
                            gender = meta.get("gender", gender)
                            region = meta.get("region", region)
                            language = meta.get("language", language)
                    except Exception as e:
                        print(f"Error reading metadata for {name}: {e}")

                self._voices[name] = Voice(
                    voice_id=name, 
                    prompt_wav=wav, 
                    prompt_text=txt,
                    name=display_name,
                    gender=gender,
                    region=region,
                    language=language
                )

    def list(self) -> List[Dict]:
        # Return list of dicts instead of list of strings
        return [
            {
                "id": v.voice_id,
                "name": v.name,
                "gender": v.gender,
                "region": v.region,
                "language": v.language
            }
            for v in self._voices.values()
        ]

    def get(self, voice_id: str) -> Voice:
        v = self._voices.get(voice_id)
        if not v:
            raise KeyError(f"voice_id '{voice_id}' not found under {self.root}")
        return v