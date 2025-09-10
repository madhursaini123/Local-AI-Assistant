import subprocess, webbrowser, pathlib, re
import os
from pathlib import Path

def open_browser(url="https://www.google.com/"):
    webbrowser.open(url)

def open_youtube(query=None):
    if query:
        webbrowser.open(f"https://www.youtube.com/results?search_query={query}")
    else:
        webbrowser.open("https://www.youtube.com/")

def open_chrome():
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    subprocess.Popen([chrome_path])

def open_file(path):
    p = pathlib.Path(path)
    if p.exists():
        os.startfile(str(p))
    else:
        print(f"[!] File not found: {p}")

COMMANDS = [
    (re.compile(r"\bopen\b.*\bchrome\b",  re.I),           lambda m: open_chrome()),
    (re.compile(r"\b(open|launch)\b.*\b(browser)\b", re.I), lambda m: open_browser()),
    (re.compile(r"\b(open|launch)\b.*\byoutube\b(?: (?P<q>.+))?", re.I),
                                                         lambda m: open_youtube(m.group("q"))),
    (re.compile(r"\bopen file\b (?P<path>.+)",  re.I),     lambda m: open_file(m.group("path"))),
]

def maybe_run_command(text: str) -> bool:
    for rgx, fn in COMMANDS:
        m = rgx.search(text)
        if m:
            fn(m)
            return True
    return False
