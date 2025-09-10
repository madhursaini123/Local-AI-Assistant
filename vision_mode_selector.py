import os
from pathlib import Path
BASE_DIR = Path("D:/python/Jarvis_AI")  # use raw string or forward slashes

face_recog_path     = BASE_DIR / "face_recognition" / "face_recog.py"
object_detect_path  = BASE_DIR / "utsiskills" / "object_detect_test.py"
emotion_path        = BASE_DIR / "utsiskills" / "emotion_test.py"
qr_code_path        = BASE_DIR / "utsiskills" / "QRcode_scanner.py"
full_system_path    = BASE_DIR / "ar_glasses_scanner.py"

print("Select AR Glasses Vision mode")
print("1. Face recongition")
print("2. Object recognition")
print("3. Emotion recognition")
print("4. QR/bar code scanning")
print("5. Run all Modules")

choice = input("Enter your choice: ")

# ✅ Execute the selected script
if choice == "1":
    os.system(f'python "{face_recog_path}"')
elif choice == "2":
    os.system(f'python "{object_detect_path}"')
elif choice == "3":
    os.system(f'python "{emotion_path}"')
elif choice == "4":
    os.system(f'python "{qr_code_path}"')
elif choice == "5":
    os.system(f'python "{full_system_path}"')
else:
    print("❌ Invalid choice. Exiting.")



    