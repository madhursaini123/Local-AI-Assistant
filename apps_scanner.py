import os
import json
from pathlib import Path

APP_DIRS = [
    r"C:\Program Files",
    r"C:\Program Files (x86)",
    r"C:\Users\Madhur2001\AppData\Local",
    r"D:\Games"
]

EXCLUDE = ["unins", "setup", "uninstall", "update", "crash", "helper", "mono", "vc_redist", "install"]
apps = {}

def is_valid_exe(filename):
    lower = filename.lower()
    return (
        lower.endswith(".exe") and
        not any(x in lower for x in EXCLUDE)
    )

for base in APP_DIRS:
    for root, dirs, files in os.walk(base):
        for file in files:
            if is_valid_exe(file):
                name = Path(file).stem.lower()
                full_path = os.path.join(root, file)
                if name not in apps:  # avoid duplicates
                    apps[name] = full_path

Path("memory").mkdir(exist_ok=True)
Path("memory/apps.json").write_text(json.dumps(apps, indent=2))
print(f"✅ Saved {len(apps)} app entries to memory/apps.json")
