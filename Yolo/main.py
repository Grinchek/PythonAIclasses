###################### Classwork #########################

# import cv2
# from ultralytics import YOLO

# imagepath = "2.jpg"

# img = cv2.imread(imagepath)

# print("This image can be readed")

# model = YOLO("yolov8n.pt")

# results= model.predict(source = imagepath, conf=0.25)

# result_martix = results[0].plot()

# cv2.imshow("Result cnowing of objects", result_martix)

# cv2.waitKey(0)

# cv2.destroyAllWindows()
###################### Classwork #########################
###################### Homework video recognizing #########################
# import cv2
# import os
# from ultralytics import YOLO

# # === Налаштування ===
# video_path = "video.mp4"     # шлях до відео
# output_folder = "images"     # куди зберігати кадри
# model_path = "yolov8n.pt"    # модель
# conf_threshold = 0.25        # поріг упевненості

# # === Ініціалізація ===
# os.makedirs(output_folder, exist_ok=True)
# model = YOLO(model_path)
# cap = cv2.VideoCapture(video_path)
# frame_count = 0
# saved_count = 0

# # === Обробка ===
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_count += 1
#     results = model.predict(frame, conf=conf_threshold, verbose=False)
#     annotated = results[0].plot()

#     # Якщо об’єкти знайдено — зберігаємо кадр
#     if len(results[0].boxes) > 0:
#         filename = f"frame_{frame_count:05d}.jpg"
#         save_path = os.path.join(output_folder, filename)
#         cv2.imwrite(save_path, annotated)
#         saved_count += 1

#     cv2.imshow("Detection", annotated)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# print(f"✅ Оброблено кадрів: {frame_count}")
# print(f"💾 Збережено кадрів із розпізнаними об'єктами: {saved_count}")
# print(f"📂 Усі зображення: {os.path.abspath(output_folder)}")
###################### Homework video recognizing #########################

###################### Homework machine learning #########################
import argparse
import os
from pathlib import Path
from ultralytics import YOLO
from dotenv import load_dotenv

# -------- Helpers --------
def try_download_with_roboflow(api_key, workspace, project, version, split="yolov8"):
    from roboflow import Roboflow  # вимагатиме pip install roboflow
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    ver = proj.version(int(version))
    dataset = ver.download(split)      # створить папку {project}-{version}/ з data.yaml
    data_yaml = Path(dataset.location) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError("Не знайдено data.yaml у завантаженому датасеті Roboflow.")
    return str(data_yaml)

def count_images_in_yolo_split(data_root: Path) -> int:
    # Порахувати картинки у images/train + images/val (або тільки train, якщо треба)
    total = 0
    for sub in ["images/train", "images/val"]:
        p = data_root / sub
        if p.exists():
            total += sum(1 for f in p.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"} )
    return total

# -------- Main --------
def main():
    load_dotenv()  # підхопить .env, якщо є

    parser = argparse.ArgumentParser(description="Auto-download dataset (Roboflow if provided, else COCO128) and train YOLOv8.")
    parser.add_argument("--epochs", type=int, default=30, help="Кількість епох (30 — ок для демо).")
    parser.add_argument("--imgsz", type=int, default=640, help="Розмір зображення.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", type=str, default="", help="'' авто | 'cpu' | '0' для GPU.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Стартова модель (n/s/m/l/x).")
    parser.add_argument("--predict_dir", type=str, default="./examples", help="Папка з тестовими зображеннями для інференсу.")
    parser.add_argument("--conf", type=float, default=0.25, help="Поріг впевненості для інференсу.")
    # Можна і через аргументи, якщо не хочеш .env
    parser.add_argument("--rf_api_key", type=str, default=os.getenv("ROBOFLOW_API_KEY"))
    parser.add_argument("--rf_workspace", type=str, default=os.getenv("ROBOFLOW_WORKSPACE"))
    parser.add_argument("--rf_project", type=str, default=os.getenv("ROBOFLOW_PROJECT"))
    parser.add_argument("--rf_version", type=str, default=os.getenv("ROBOFLOW_VERSION"))
    args = parser.parse_args()

    # 1) Визначаємо джерело даних
    use_roboflow = all([args.rf_api_key, args.rf_workspace, args.rf_project, args.rf_version])

    if use_roboflow:
        print("[INFO] Виявлено параметри Roboflow: завантажую датасет...")
        data_yaml = try_download_with_roboflow(
            api_key=args.rf_api_key,
            workspace=args.rf_workspace,
            project=args.rf_project,
            version=args.rf_version,
        )
        data_root = Path(data_yaml).parent
        total_imgs = count_images_in_yolo_split(data_root)
        print(f"[INFO] Знайдено зображень у train+val: {total_imgs}")
        if total_imgs < 100:
            print("[WARN] У датасеті < 100 зображень. Для вимоги мінімум 100 краще додати даних або оновити версію в Roboflow.")
    else:
        # 2) Фолбек: COCO128 — Ultralytics сам завантажить при тренуванні
        print("[INFO] Параметри Roboflow не задані. Використовую COCO128 (буде завантажено автоматично).")
        # Ultralytics вже знає шлях до coco128.yaml всередині пакету
        # модель сама підтягне архів при першому train
        data_yaml = "coco128.yaml"

    # 3) Навчання
    model = YOLO(args.model)
    print("[INFO] Початок тренування...")
    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs/train",
        name="auto_dataset_yolov8"
    )

    # 4) Валідація
    print("[INFO] Валідація...")
    model.val(data=data_yaml, imgsz=args.imgsz, device=args.device)

    # 5) Інференс (якщо є приклади)
    pred_dir = Path(args.predict_dir)
    if pred_dir.exists():
        print(f"[INFO] Інференс на {pred_dir} ...")
        model.predict(
            source=str(pred_dir),
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            save=True,
            project="runs/predict",
            name="auto_dataset_yolov8"
        )
        print("[OK] Результати інференсу: runs/predict/auto_dataset_yolov8")
    else:
        print(f"[WARN] Папку з прикладами не знайдено ({pred_dir}). Пропускаю інференс.")

    print("[DONE] Готово! Найкращі ваги: runs/train/auto_dataset_yolov8/weights/best.pt")


if __name__ == "__main__":
    main()


# ================= ЗВІТ =================
# Програма автоматично завантажує навчальні дані (датасет COCO128 або власний набір з Roboflow)
# і навчає модель YOLOv8 для розпізнавання об’єктів на зображеннях.

# Під час виконання:
# - Відбувається завантаження або створення датасету з понад 100 зображень.
# - Модель проходить процес навчання протягом заданої кількості епох.
# - Після завершення формується готова навчена модель (best.pt), 
#   яка здатна розпізнавати об’єкти на нових зображеннях.
# - У папці runs/train/... зберігаються графіки, звіти та ваги моделі.
# - Програма також проводить тестове розпізнавання (інференс) на прикладних зображеннях 
#   і зберігає результати у папці runs/predict/...

# Отже, в результаті роботи ми отримуємо:
# 1. Навчену нейронну мережу YOLOv8 (best.pt);
# 2. Графіки зміни точності, повноти та втрат під час тренування;
# 3. Зображення з розпізнаними об’єктами як демонстрацію роботи моделі.

# Програма демонструє повний цикл роботи з нейронною мережею:
# від завантаження даних і навчання — до отримання готової моделі, здатної розпізнавати об’єкти.
# ========================================

###################### Homework machine learning #########################


