import importlib.util
import os
from pathlib import Path
from utsiskills.vision import open_webcam

SKILLS_FOLDER = Path(r"D:\python\Jarvis_AI\utsiskills")
loaded_skills = []

def load_skills():
    for file in SKILLS_FOLDER.glob("*.py"):
        spec = importlib.util.spec_from_file_location(file.stem, file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "can_handle") and hasattr(module, "handle"):
            loaded_skills.append(module)

def run_skill_if_match(text: str) -> bool:
    for skill in loaded_skills:
        if skill.can_handle(text):
            skill.handle(text)
            return True
        elif "open webcam" in text or "start camera" in text:
            open_webcam()
            return True
    return False

# inside run_skill_if_match(text)

