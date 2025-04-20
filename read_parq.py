import pandas as pd

   # Укажи путь к твоему файлу .parquet
file_path = 'data/07_model_output/predictions.parquet'

   # Загрузка данных из файла .parquet
df = pd.read_parquet(file_path)

   # Вывод информации о DataFrame
print("Информация о DataFrame:")
print(df.info())

   # Вывод первых нескольких строк данных
print("\nПервые 5 строк данных:")
print(df.head())