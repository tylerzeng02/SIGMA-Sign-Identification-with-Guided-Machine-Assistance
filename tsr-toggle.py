from gtts import gTTS
import os
import tempfile
import time
import cv2
from ultralytics import YOLO
import playsound

model = YOLO("tsr-v3.pt")
cap = cv2.VideoCapture(0)

last_spoken = {}
cooldown = 3
language = 'en'  # Default to English

# Language-specific full phrases
translations_all = {
    'en': {
        "green": "green light",
        "red": "red light",
        "yellow": "yellow light",
        "school-zone": "school zone",
        "pedestrian-crossing": "pedestrian crossing",
        "construction-sign": "construction sign"
    },
    'ko': {
        "green": "초록 신호",
        "red": "빨간 신호",
        "yellow": "노란 신호",
        "school-zone": "학교 구역",
        "pedestrian-crossing": "횡단보도",
        "construction-sign": "공사 표지판"
    },
    'ja': {
        "green": "青信号",
        "red": "赤信号",
        "yellow": "黄信号",
        "school-zone": "通学路",
        "pedestrian-crossing": "横断歩道",
        "construction-sign": "工事標識"
    },
    'es': {
        "green": "luz verde",
        "red": "luz roja",
        "yellow": "luz amarilla",
        "school-zone": "zona escolar",
        "pedestrian-crossing": "paso de peatones",
        "construction-sign": "señal de construcción"
    },
    'fr': {
        "green": "feu vert",
        "red": "feu rouge",
        "yellow": "feu jaune",
        "school-zone": "zone scolaire",
        "pedestrian-crossing": "passage piéton",
        "construction-sign": "panneau de chantier"
    },
    'zh-CN': {
        "green": "绿灯",
        "red": "红灯",
        "yellow": "黄灯",
        "school-zone": "学校区域",
        "pedestrian-crossing": "人行横道",
        "construction-sign": "施工标志"
    },
    'pa': {
        "green": "ਹਰੀ ਬੱਤੀ",
        "red": "ਲਾਲ ਬੱਤੀ",
        "yellow": "ਪੀਲੀ ਬੱਤੀ",
        "school-zone": "ਸਕੂਲ ਜ਼ੋਨ",
        "pedestrian-crossing": "ਪੈਦਲ ਚਲਣ ਵਾਲਿਆਂ ਦੀ ਪਾਰਿੰਗ",
        "construction-sign": "ਨਿਰਮਾਣ ਚਿੰਨ੍ਹ"
    },
    'la': {
        "green": "lux viridis",
        "red": "lux rubra",
        "yellow": "lux flava",
        "school-zone": "regio scholae",
        "pedestrian-crossing": "transitus pedestris",
        "construction-sign": "signum constructionis"
    }
}

def speak(text, lang='en'):
    with tempfile.NamedTemporaryFile(delete=True, suffix='.mp3') as fp:
        tts = gTTS(text=text, lang=lang if lang != 'la' else 'en')
        tts.save(fp.name)
        playsound.playsound(fp.name)

lang_names = {
    'en': 'English',
    'ko': 'Korean',
    'ja': 'Japanese',
    'es': 'Spanish',
    'fr': 'French',
    'zh-CN': 'Mandarin',
    'pa': 'Punjabi',
    'la': 'Latin'
}

print("Starting multilingual sign detection...")
print("Press key to change language:")
print("e=English, k=Korean, j=Japanese, s=Spanish, f=French, c=Mandarin, p=Punjabi, l=Latin, q=Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()
    current_time = time.time()

    translations = translations_all.get(language, translations_all['en'])

    boxes = results[0].boxes
    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            translated = translations.get(label, label)

            if label not in last_spoken or (current_time - last_spoken[label]) > cooldown:
                print(f"Detected: {label} → Speaking: {translated} ({lang_names.get(language, language)})")
                speak(translated, lang=language)
                last_spoken[label] = current_time

    lang_text = f"Language: {lang_names.get(language, language)}"
    cv2.putText(annotated_frame, lang_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Real-Time Sign Detector", annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('e'):
        language = 'en'
    elif key == ord('k'):
        language = 'ko'
    elif key == ord('j'):
        language = 'ja'
    elif key == ord('s'):
        language = 'es'
    elif key == ord('f'):
        language = 'fr'
    elif key == ord('c'):
        language = 'zh-CN'
    elif key == ord('p'):
        language = 'pa'
    elif key == ord('l'):
        language = 'la'

cap.release()
cv2.destroyAllWindows()
