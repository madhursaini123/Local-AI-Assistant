import os
from utsiskills.utils import speak

def can_handle(text: str) -> bool:
    return "open file" in text.lower()

def handle(text: str):
    try:
        filename = text.split("open file")[-1].strip()
        path = os.path.abspath(filename)
        if os.path.exists(path):
            os.startfile(path)
            speak(f"Opening {filename}")
        else:
            speak("I couldn't find that file.")
    except Exception as e:
        speak("Something went wrong.")
        print("File error:", e)
