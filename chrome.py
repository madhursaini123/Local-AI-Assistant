import subprocess

def can_handle(text: str) -> bool:
    return "open chrome" in text.lower() or "launch chrome" in text.lower()

def handle(text: str):
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    subprocess.Popen([chrome_path])
