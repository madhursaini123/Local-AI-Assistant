import re
import requests
from utsiskills.utils import speak  # ✅ This will now always work

def can_handle(text: str) -> bool:
    return "weather" in text.lower() or "temperature" in text.lower()

def handle(text: str):
    match = re.search(r"in (\w+(?: \w+)?)", text.lower())
    city = match.group(1) if match else "Delhi"

    url = f"https://wttr.in/{city}?format=3"
    try:
        res = requests.get(url)
        if res.status_code == 200:
            print(res.text)
            speak(res.text)
        else:
            speak("Could not fetch weather information.")
    except Exception as e:
        speak("An error occurred while getting the weather.")
        print("Error:", e)
