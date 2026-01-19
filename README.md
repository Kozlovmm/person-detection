# Детекция людей в толпе

Скрипт на Python запускает модель для детекции людей на видео и сохраняет размеченный MP4 через OpenCV. Работает на Linux, macOS и Windows.

## Установка
1. `python3 -m venv .venv`
2. `source .venv/bin/activate` (Linux/macOS) или `.venv\Scripts\activate` (Windows)
3. `pip install -r requirements.txt`

## Запуск
1. Поместите входные видео в папку `assets/`.
2. Выполните:
   ```bash
   python -m src.main
   ```


### Параметры
- `--model`: путь к весам (по умолчанию `yolov8n.pt`)
- `--conf`: порог уверенности (по умолчанию `0.25`) (Для видео с большим количеством людей рекомендуется 0.15 - 0.25, Для видео с небольшим количеством 0.35 - 0.5)
- `--imgsz`: размер инференса (по умолчанию размер входного видео)
- `--device`: устройство инференса (`auto`, `cpu`, `cuda:0`, `mps`) (по умолчанию 'auto')


## Структура проекта
- `src/main.py` — запускает пайплайн и сохраняет метрики
- `src/detector.py` — обёртка над моделью и результат детекции
- `src/draw.py` — Графика
- `src/video_io.py` — чтение метаданных и сохранения обработанного видео
- `assets/` — входные видео
- `outputs/` — размеченные видео и метрики

## Возможные улучшения
- Потенциальные сложности: окклюзии, очень маленькие/далёкие люди, смазанные
  кадры, быстрые движения, фото людей/другие объекты.
- Как улучшать:
  1. Использовать более мощную модель
  2. Добавить трекинг для стабильных ID
  3. Дообучить на похожих данных
  4. Настроить постобработку (порог/площадь бокса, сглаживание координат)
  5. Фильтровать слишком маленькие боксы и резкие скачки размера
  6. Убирать вспышки: удалять объекты, появляющиеся на 1 кадре
  7. Использовать разные пороги для разных зон изображения
  8. Прогонять с разными масштабами/порогами уверенности и объединять результаты
  9. Добавить предварительную обработку кадров (контраст, шумоподавление)


# Detecting people in a crowd

Python script that runs a pretrained model to detect people on video, draws
thin boxes, and saves an annotated MP4 via OpenCV. Works on Linux, macOS, and
Windows.

## Setup
1. `python3 -m venv .venv`
2. `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\activate`
   (Windows)
3. `pip install -r requirements.txt`

## Run
1. Put input videos in `assets/`.
2. Run:
```bash
python -m src.main
```

### Options
- `--model`: weights path (default `yolov8n.pt`)
- `--conf`: confidence threshold (default `0.25`)
- `--imgsz`: inference size (default input video size)
- `--device`: inference device (`auto`, `cpu`, `cuda:0`, `mps`; default `auto`)

## Project structure
- `src/main.py` - pipeline entry and metrics
- `src/detector.py` - model wrapper and detection output
- `src/draw.py` - drawing utilities
- `src/video_io.py` - metadata reading and video saving
- `assets/` - input videos
- `outputs/` - annotated videos and metrics

## Improvements
- Issues: occlusions, very small/far people, blur, fast motion, photos/posters.
- Improve by:
  1. Larger model
  2. Add tracking for stable IDs
  3. Fine-tune on similar data
  4. Tune post-processing (threshold/box area, smoothing)
  5. Filter tiny boxes and sudden size jumps
  6. Drop one-frame flickers
  7. Different thresholds per image zone
  8. Multi-scale/threshold runs with fusion
  9. Preprocess frames (contrast, denoise)
