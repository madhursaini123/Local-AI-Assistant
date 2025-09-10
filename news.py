from datetime import datetime, timedelta
import re
import requests
from utsiskills.utils import speak

API_KEY = "af62c93eb9d67eadaa62cdde6f847c08"

def extract_category(text):
    categories = ["technology", "business", "sports", "health", "science", "entertainment"]
    for cat in categories:
        if cat in text.lower():
            return cat
    return None

def can_handle(text: str) -> bool:
    return any(word in text.lower() for word in ["news", "headlines"])

def handle(text: str):
    base_url = f"http://api.mediastack.com/v1/news?access_key={API_KEY}"
    params = {
        "countries": "in, us, gb, de",
        "limit": 5
    }

    if "international" in text.lower():
        params["countries"] = "us,gb,de"

    if "yesterday" in text.lower():
        yest = datetime.now() - timedelta(days=1)
        params["date"] = yest.strftime('%Y-%m-%d')

    category = extract_category(text)
    if category:
        params["categories"] = category

    try:
        res = requests.get(base_url, params=params)
        data = res.json()
        articles = data.get("data", [])

        if not articles:
            speak("Sorry, I couldn't find any news matching your request.")
            return

        speak("Here are the top headlines:")
        for i, article in enumerate(articles, 1):
          title = article.get("title", "No title")
          summary = article.get("description", "")
    
          print(f"{i}. {title}")
          if summary:
            print(f"   ↪️ {summary}")
            speak(f"{title}. {summary}")
          else:
            speak(title)

    except Exception as e:
        print("News fetch error:", e)
        speak("Sorry, I couldn't fetch the news right now.")
