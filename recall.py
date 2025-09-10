# recall.py
def can_handle(text: str) -> bool:
    return "remember" in text.lower() or "recall" in text.lower()

def handle(text: str):
    from brain.intelligence import recall_memory
    from utsiskills.utils import speak
    
    words = text.split()
    keyword = words[-1] if len(words) > 1 else " "
    result = recall_memory(keyword)
    print("🧠 Recall:", result)
    speak(result)
