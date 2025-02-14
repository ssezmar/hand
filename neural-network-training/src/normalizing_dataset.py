import numpy as np
import requests
import json
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import os
import pandas as pd

# COCO API URL (можно заменить на локальный путь)
COCO_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_ANNOTATIONS_PATH = "../data/person_keypoints_train2017.json"  # Локальный путь после загрузки

# Ensure the file exists
if not os.path.exists(COCO_ANNOTATIONS_PATH):
    raise FileNotFoundError(f"File not found: {COCO_ANNOTATIONS_PATH}")

# Загружаем аннотации COCO
coco = COCO(COCO_ANNOTATIONS_PATH)

# Ключевые точки тела
KEYPOINTS = {"left_shoulder": 5, "right_shoulder": 6,
             "left_elbow": 7, "right_elbow": 8,
             "left_wrist": 9, "right_wrist": 10}

# Функция нормализации координат
def normalize_coords(coords):
    coords = np.array(coords, dtype=np.float32)
    min_val, max_val = np.min(coords, axis=0), np.max(coords, axis=0)
    return (coords - min_val) / (max_val - min_val)

# Функция вычисления угла в локте
def elbow_angle(shoulder, elbow, wrist):
    vec1 = np.array(elbow) - np.array(shoulder)
    vec2 = np.array(wrist) - np.array(elbow)
    cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi
    return angle

# Создаем DataFrame для хранения данных
columns = ['shoulder_x', 'shoulder_y', 'elbow_x', 'elbow_y', 'wrist_x', 'wrist_y', 'angle', 'arm_state']
df = pd.DataFrame(columns=columns)

# Выбираем изображения с людьми
image_ids = coco.getImgIds(catIds=[1])  # ID людей

for image_id in image_ids:
    image_data = coco.loadImgs(image_id)[0]
    annotation_ids = coco.getAnnIds(imgIds=image_data['id'], catIds=[1])
    annotations = coco.loadAnns(annotation_ids)

    # Извлекаем координаты ключевых точек
    for ann in annotations:
        keypoints = ann["keypoints"]
        left_shoulder = keypoints[KEYPOINTS["left_shoulder"] * 3:KEYPOINTS["left_shoulder"] * 3 + 2]
        left_elbow = keypoints[KEYPOINTS["left_elbow"] * 3:KEYPOINTS["left_elbow"] * 3 + 2]
        left_wrist = keypoints[KEYPOINTS["left_wrist"] * 3:KEYPOINTS["left_wrist"] * 3 + 2]

        # Нормализуем координаты
        norm_coords = normalize_coords([left_shoulder, left_elbow, left_wrist])

        # Вычисляем угол
        angle = elbow_angle(norm_coords[0], norm_coords[1], norm_coords[2])
        arm_state = "Согнута" if angle < 150 else "Разогнута"

        # Добавляем данные в DataFrame
        df = df.append({
            'shoulder_x': norm_coords[0][0],
            'shoulder_y': norm_coords[0][1],
            'elbow_x': norm_coords[1][0],
            'elbow_y': norm_coords[1][1],
            'wrist_x': norm_coords[2][0],
            'wrist_y': norm_coords[2][1],
            'angle': angle,
            'arm_state': arm_state
        }, ignore_index=True)

        # Визуализируем
        plt.scatter(norm_coords[:, 0], norm_coords[:, 1], color=['r', 'g', 'b'])
        plt.plot(norm_coords[:, 0], norm_coords[:, 1], 'k-')
        plt.title(f"Угол: {angle:.2f}° ({arm_state})")
        plt.show()
        break  # Только одно изображение

# Сохраняем DataFrame в CSV файл
df.to_csv('../data/normalized_dataset.csv', index=False)
