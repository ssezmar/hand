import pandas as pd

# Загрузка датасета
df = pd.read_csv('./data/dataset.csv')

# Проверка наличия столбца 'label'
if 'label' not in df.columns:
    raise KeyError("Столбец 'label' отсутствует в датасете")

# Проверка полноты датасета
total_samples = len(df)
bent_arms = df[df['label'] == 1].shape[0]
extended_arms = df[df['label'] == 0].shape[0]

# Вывод результатов
print(f'Total samples: {total_samples}')
print(f'Bent arms: {bent_arms}')
print(f'Extended arms: {extended_arms}')