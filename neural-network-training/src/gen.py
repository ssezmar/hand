import numpy as np
import pandas as pd

# Количество образцов для генерации
num_samples_per_class = 10134
batch_size = 10134  # Размер порции для сохранения

# Функция для генерации случайных 3D координат
def generate_3d_point():
    return np.random.uniform(-1, 1, 3)

# Функция для определения, согнута ли рука
def is_bent(shoulder, elbow, wrist):
    upper_arm = np.linalg.norm(elbow - shoulder)
    forearm = np.linalg.norm(wrist - elbow)
    total_length = np.linalg.norm(wrist - shoulder)
    return 1 if total_length < (upper_arm + forearm) * 0.95 else 0

# Создание DataFrame для хранения данных
columns = ['shoulder_x', 'shoulder_y', 'shoulder_z', 
           'elbow_x', 'elbow_y', 'elbow_z', 
           'wrist_x', 'wrist_y', 'wrist_z', 
           'label']
df = pd.DataFrame(columns=columns)

# Генерация датасета и сохранение порциями
data = []
count_bent = 0
count_extended = 0

while count_bent < num_samples_per_class or count_extended < num_samples_per_class:
    shoulder = generate_3d_point()
    elbow = generate_3d_point()
    wrist = generate_3d_point()
    label = is_bent(shoulder, elbow, wrist)
    
    if label == 1 and count_bent < num_samples_per_class:
        data.append(np.concatenate([shoulder, elbow, wrist, [label]]))
        count_bent += 1
    elif label == 0 and count_extended < num_samples_per_class:
        data.append(np.concatenate([shoulder, elbow, wrist, [label]]))
        count_extended += 1
    
    # Сохранение данных порциями
    if len(data) >= batch_size:
        df = pd.DataFrame(data, columns=columns)
        df.to_csv('./data/dataset.csv', mode='a', header=not pd.read_csv('./data/dataset.csv').empty, index=False)
        data = []

# Сохранение оставшихся данных
if data:
    df = pd.DataFrame(data, columns=columns)
    df.to_csv('./data/dataset.csv', mode='a', header=not pd.read_csv('./data/dataset.csv').empty, index=False)

print("Датасет успешно сгенерирован и сохранен в файл 'dataset.csv'")