import re
import json
from datetime import datetime
from pathlib import Path
from dateutil import parser as dateparser
from utsiskills.utils import speak

REMINDER_FILE = Path("memory/reminders.json")
REMINDER_FILE.parent.mkdir(exist_ok=True)

# Load reminders
if REMINDER_FILE.exists():
    reminders = json.loads(REMINDER_FILE.read_text(encoding="utf-8"))
else:
    reminders = []

def save_reminders():
    REMINDER_FILE.write_text(json.dumps(reminders, indent=2, ensure_ascii=False), encoding="utf-8")

def can_handle(text: str) -> bool:
    return any(x in text.lower() for x in [
        "remind me", "add a task", "what are my reminders", "what's on my schedule"
    ])

def handle(text: str):
    lower = text.lower()
    now = datetime.now()

    if "remind me" in lower or "add a task" in lower:
        try:
            action = re.sub(r"(remind me to|add a task to)", "", text, flags=re.I).strip()
            date_match = dateparser.parse(action, fuzzy=True)

            description = re.sub(str(date_match), "", action, flags=re.I).strip()
            if not description:
                description = "Unnamed task"

            reminders.append({
                "task": description,
                "time": str(date_match),
                "done": False
            })
            save_reminders()
            speak(f"Okay, I will remind you to {description} at {date_match.strftime('%I:%M %p on %B %d')}")
        except Exception as e:
            speak("Sorry, I couldn't understand the reminder.")
            print("Reminder parse error:", e)

    elif "reminders" in lower or "schedule" in lower:
        today = now.date()
        todays = [r for r in reminders if dateparser.parse(r["time"]).date() == today and not r.get("done")]
        if not todays:
            speak("You have no reminders today.")
        else:
            speak("Here are your reminders for today:")
            for r in todays:
                t = dateparser.parse(r["time"])
                print(f"🔔 {t.strftime('%I:%M %p')} - {r['task']}")
                speak(f"At {t.strftime('%I:%M %p')}, {r['task']}")

def check_upcoming_reminders():
    now = datetime.now()
    for r in reminders:
        try:
            if r.get("done"):
                continue
            r_time = dateparser.parse(r["time"])
            if -30 <= (r_time - now).total_seconds() <= 60:
                speak(f"Reminder: {r['task']} at {r_time.strftime('%I:%M %p')}")
                print(f"🔔 Reminder: {r['task']} at {r_time.strftime('%I:%M %p')}")
                r["done"] = True
        except Exception as e:
            print("Reminder check error:", e)
            continue
    save_reminders()
