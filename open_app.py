import json
from pathlib import Path
import subprocess
import difflib
from utsiskills.utils import speak
import os

APPS_PATH = Path("memory/apps.json")

def load_apps():
    if APPS_PATH.exists():
        return json.loads(APPS_PATH.read_text(encoding="utf-8"))
    return {}

def fuzzy_find(app_name, app_dict):
    matches = difflib.get_close_matches(app_name.lower(), app_dict.keys(), n=1, cutoff=0.5)
    return matches[0] if matches else None

def can_handle(text: str) -> bool:
    return "open" in text.lower()

def handle(text: str):
    apps = load_apps()
    app_name = text.lower().replace("open", "").strip()

    match = fuzzy_find(app_name, apps)
    if match:
        app_path = apps[match]
        if os.path.exists(app_path):
            speak(f"Opening {match}")
            subprocess.Popen([app_path])
        else:
            speak(f"I found {match}, but its file path doesn't exist anymore.")
    else:
        speak("Sorry, I couldn't find that app. Try updating the app list.")
