# CV Demo · комп'ютерний зір для конференції

Локальний веб-додаток для демонстрації comp-vision на стенді: ноутбук + USB-камера + 65" ТВ. Браузер у повноекранному режимі — і ви маєте сучасний інтерфейс із бічною панеллю режимів, де відвідувачі бачать себе на екрані з накладеними детекціями.

## Можливості

| Режим | Що робить |
|---|---|
| **Детекція людей** | Кожна людина — у власному кольоровому bounding-боксі (без червоного). |
| **Зони інтересу** | Малюйте полігональні зони мишею. Коли хтось у зоні — її периметр і бокс людини стають червоними. |
| **Засоби захисту** | Детектує шолом на голові. Зелений бокс + іконка шолома, якщо є. Червоний бокс + перекреслена іконка, якщо немає. |
| **Профілювання** | Стать (чоловік/жінка), вікова група (8 діапазонів від 0 до 60+) і настрій (нормальний / веселий / сумний / сердитий / наляканий / здивований / роздратований). Колір боксу залежить від настрою. |

## Стек

- **Python + Flask** — сервер, MJPEG-стрім, REST для перемикання режимів.
- **OpenCV** — захоплення з вебкамери, YuNet face detection, FER+ emotion (ONNX), Levi-Hassner gender + age (Caffe). Все через `cv2.dnn`, без TensorFlow.
- **Ultralytics YOLOv8 (nano)** — детекція людей.
- **`keremberke/yolov8n-hard-hat-detection`** (HuggingFace) — детекція шолома.
- **HTML5 + JS** — інтерфейс із canvas-overlay для малювання зон.

## Вимоги

- Windows 10/11 (працює також на macOS/Linux).
- **Python 3.10 — 3.14**. Раніше використовувалась DeepFace + TensorFlow і потрібен був саме 3.11 — тепер залежності легші (`onnxruntime` замість TF), тому 3.14 теж працює.
- USB- або вбудована вебкамера.
- ~1 ГБ вільного місця (моделі завантажуються при першому запуску).

### Перевірка версії Python

```powershell
py --list      # покаже всі встановлені версії
py -3.11 -V    # має вивести: Python 3.11.x
```

Якщо `Python 3.11` немає у списку — поставте його. **Адмін-прав не потрібно**, є два варіанти.

#### Варіант A. Інсталятор python.org (per-user)

1. Завантажте: https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
2. Запустіть. На першому екрані:
   - ✅ **Add python.exe to PATH**
   - ❌ **Install for all users** — НЕ ставте (саме ця галочка вимагає адмінки)
3. Натисніть **Install Now** — установка в `%LOCALAPPDATA%\Programs\Python\Python311\` (без адмінки).

#### Варіант B. uv (рекомендую)

[`uv`](https://github.com/astral-sh/uv) — швидкий менеджер Python + пакетів від Astral. Не потребує адмінки, сам завантажує потрібну версію Python у профіль користувача, а пакети ставить у 5–10 разів швидше за pip.

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# перезапустіть PowerShell після інсталяції
uv python install 3.11
```

> Не видаляйте Python 3.14 — і `py -3.11`, і `uv` спокійно існують паралельно.

## Встановлення (Windows)

### З `uv` (швидше)

```powershell
cd cv-demo
uv venv --python 3.11 .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

### Зі стандартним `pip`

```powershell
cd cv-demo
py -3.11 -m venv .venv
.venv\Scripts\activate

python -V                                   # перевірте: Python 3.11.x
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> На macOS/Linux замість `py -3.11` використовуйте `python3.11`.

### Якщо все одно бачите помилку компіляції NumPy

Це означає, що venv створено не на 3.11. Видаліть `.venv` і повторіть:
```powershell
Remove-Item -Recurse -Force .venv
py -3.11 -m venv .venv
.venv\Scripts\activate
python -V         # переконайтеся: 3.11.x
pip install -r requirements.txt
```

## Запуск

```powershell
python app.py
```

Відкрийте браузер на ноутбуку, який підключено до ТВ:

```
http://localhost:5000
```

Натисніть **F11** для повноекранного режиму на 65".

### Перший запуск

При першому використанні режимів додаток завантажує моделі в `./models/`:

- `yolov8n.pt` (~6 МБ) — детекція людей;
- `yolov8n-hardhat.pt` (~6 МБ) — детекція шолома;
- `yunet.onnx` (~230 КБ) — детектор облич (профілювання);
- `emotion-ferplus-8.onnx` (~35 МБ) — емоції;
- `gender_net.caffemodel` (~45 МБ) — стать.

Перший раз режим увімкнеться з затримкою 5–20 с — це нормально.

### ⚠️ Ручне завантаження моделей (за корпоративним проксі)

Корпоративні мережі часто перехоплюють HTTPS своїм CA-сертифікатом — тоді або відмовляє SSL-валідація Python, або проксі повертає HTML-сторінку блокування замість бінарних ваг (тоді ви побачите в логах щось на кшталт `PytorchStreamReader failed reading zip archive — file is corrupted`).

Найнадійніший спосіб — завантажити файли з домашнього/мобільного інтернету і покласти їх у папку `models/` поряд із `app.py`. Створіть папку `models/` якщо її немає, і збережіть туди шість файлів:

| Файл (саме таке ім'я) | Розмір | Посилання |
|---|---|---|
| `yolov8n.pt` | ~6 МБ | https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt |
| `yolov8n-hardhat.pt` | ~6 МБ | https://huggingface.co/keremberke/yolov8n-hard-hat-detection/resolve/main/best.pt |
| `yunet.onnx` | ~230 КБ | https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx |
| `emotion-ferplus-8.onnx` | ~35 МБ | https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx |
| `gender_deploy.prototxt` | ~3 КБ | https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/gender_deploy.prototxt |
| `gender_net.caffemodel` | ~45 МБ | https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/gender_net.caffemodel |
| `age_deploy.prototxt` | ~3 КБ | https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/age_deploy.prototxt |
| `age_net.caffemodel` | ~45 МБ | https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/age_net.caffemodel |

> ❗ Файл із HuggingFace (`best.pt`) **обов'язково перейменуйте** на `yolov8n-hardhat.pt`. Усі інші файли мають зберігатися із тими самими іменами, що в URL.

Структура має виглядати так:

```
cv-demo/
├── app.py
├── models/
│   ├── yolov8n.pt
│   ├── yolov8n-hardhat.pt
│   ├── yunet.onnx
│   ├── emotion-ferplus-8.onnx
│   ├── gender_deploy.prototxt
│   └── gender_net.caffemodel
└── …
```

Як тільки файли є локально, додаток у мережу не лізе взагалі.

#### Якщо файл уже завантажився, але "битий"

Симптом — в логах постійно `PytorchStreamReader failed reading zip archive` або `file is corrupted`. На старті додаток автоматично перевіряє файли в `models/` за магічними байтами і видаляє ті, що насправді є HTML-сторінкою. Якщо хочете перевірити вручну — просто видаліть `models/` і покладіть файли заново.

## Користування

- **Перемикач режимів** справа: клік по картці.
- **Зони інтересу**: оберіть режим → `+ нова зона` → клікайте по відео для додавання точок → `завершити зону` (або подвійний клік / Enter). Можна намалювати декілька зон. `очистити все` — видалити всі.
  - `Ctrl+Z` — прибрати останню точку.
  - `Esc` — скасувати поточну зону.
- **Профілювання** оновлюється приблизно двічі на секунду (DeepFace важка) — це навмисно, щоб не падав FPS.

## Продуктивність

Архітектура вже оптимізована для CPU-демо: вся важка інференс-робота (YOLO person, YOLO helmet, face analyzer) виконується в **окремому потоці**, а MJPEG-стрім малює тільки кешовані результати. Тому FPS відеостріму обмежений лише швидкістю промальовки + JPEG-енкодом, а не моделями. Boxes лагнуть вхідний кадр приблизно на одну ітерацію інференсу (~50-80 мс) — на демо непомітно.

Тонке налаштування через змінні оточення (PowerShell):

| Змінна | Стандарт | Що робить |
|---|---|---|
| `YOLO_IMGSZ` | `384` | Вхідне розширення YOLO. Менше = швидше. `320` ~+30% FPS, `256` ще швидше але починає пропускати малі обличчя/людей. |
| `EMOTION_DEBUG` | `0` | `1` друкує топ-3 ймовірностей емоцій + зберігає кропи облич у `./debug/`. |
| `EMOTION_NEUTRAL_THRESHOLD` | `0.55` | Ймовірність "neutral", нижче якої ми пропускаємо її і беремо найкращу з решти класів. Знизити до `0.45` для ще більш динамічної реакції; підняти до `0.7` якщо модель надто часто скаче на не-нейтральні емоції. |
| `CV_CAM_WIDTH` | `1280` | Ширина захоплення з камери (не впливає на інференс — модель має свій `imgsz`). |
| `CV_CAM_HEIGHT` | `720` | Висота захоплення. Можна знизити до 960×540 щоб трохи прискорити JPEG-енкод. |
| `CV_CAM_INDEX` | *(auto)* | Конкретний індекс відеопристрою (`0`, `1`, …). Якщо не задано — система при старті сама знайде USB-камеру (вищий індекс) і обере її, інакше візьме вбудовану на `0`. |

Приклад максимально швидкого режиму:
```powershell
$env:YOLO_IMGSZ = "320"
$env:CV_CAM_WIDTH = "960"
$env:CV_CAM_HEIGHT = "540"
python app.py
```

На Ryzen 5 7535U з типовими налаштуваннями має бути 18–25 FPS у режимі "Детекція людей", 15–20 FPS у "Засобах захисту" (дві YOLO), і 18–22 FPS у "Профілюванні" (face analyzer оновлюється раз на ~0.4 с і не блокує стрім).

Якщо у вас є NVIDIA GPU + CUDA — `ultralytics` сам перемкнеться на нього і FPS виросте у 3–5 разів.

## Структура

```
cv-demo/
├── app.py              ← Flask + захоплення з камери + MJPEG
├── detector.py         ← усі моделі та режими
├── requirements.txt
├── templates/index.html
└── static/
    ├── style.css       ← темна "glass" тема
    └── app.js          ← перемикання режимів + canvas-малювання зон
```

## Налагодження детекції емоцій

Емоції тримаються на FER+ — найдоступнішому ONNX-класифікаторі, але він має сильний bias на "neutral". Тому ми застосовуємо anti-neutral threshold (`EMOTION_NEUTRAL_THRESHOLD`, дефолт 0.55): якщо ймовірність neutral нижче порогу — береться найкраща з решти емоцій. На демо-стенді це дає дуже живу реакцію навіть на слабкі вирази.

Якщо щось не так:

```powershell
$env:EMOTION_DEBUG = "1"
python app.py
```

У консолі побачите кожні ~0.4 с топ-3 ймовірностей з вибраним класом:
```
[emotion@(640,360)] neutral=0.42  happiness=0.31  sadness=0.11  →  happiness
```

— тут `neutral` був топ, але нижче 0.55, тому переможець `happiness`. У `./debug/face_*.jpg` зберігаються кропи облич, які йдуть у модель.

Якщо хочете повернути "класичну" поведінку без зсуву (тільки argmax), поставте `$env:EMOTION_NEUTRAL_THRESHOLD = "0"`.

## Поради для стенду

- Підключіть ТВ як **другий екран**, перетягніть туди браузер, F11.
- Поставте камеру на рівні очей за 2–3 м від людей — і YOLO, і обличчя будуть стабільнішими.
- Ввімкніть live-mode (звичайна детекція) на стартовому екрані — це найшвидший і найбільш "вау" режим для перших гостей.

### USB-камера

При старті додаток автоматично сканує всі відеопристрої (індекси 0–4 на Windows) і обирає **зовнішню USB-камеру** (вищий індекс), якщо вона є. Якщо USB не підключено — використовується вбудована веб-камера ноутбука (`0`). У консолі побачите щось на кшталт:

```
[camera] probing video device indices…
[camera]   index 0: ok (1280×720)
[camera]   index 1: ok (1920×1080)
[camera] using USB camera at index 1 (also saw: 0)
```

Якщо потрібно примусово зафіксувати конкретну камеру (наприклад, у вас є дві USB-камери і ви хочете першу):

```powershell
$env:CV_CAM_INDEX = "1"
python app.py
```

> Підключайте USB-камеру **до запуску** додатка. Гарячі підключення під час роботи app не перепідхоплює — потрібен перезапуск.
