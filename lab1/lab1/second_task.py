import pandas as pd

"""
Створення DataFrame за допомогою словника (dictionary)
"""
data = {
    "name": ["Olia", "Ivanna", "Anastasia", "Misha"],
    "age": [20, 19, 20, 20],
    "city": ["Koziatyn", "Kyiv", "Kyiv", "Dnipro"],
}

df = pd.DataFrame(data)
print(df.head())  # перегляд перших записів

"""
Створення DataFrame за допомогою списку (list)
"""
data = [
    ["Kyiv", 1, "Ukraine"],
    ["Berlin", 2, "Germany"],
    ["Istanbul", 3, "Turkey"],
    ["Warsaw", 4, "Poland"],
]

df = pd.DataFrame(data, columns=["city", "id", "country"])
print(df.describe())  # перегляд описової статистики


"""
Створення DataFrame за допомогою CSV файлу
"""
df = pd.read_csv("data.csv")
print(df)
print(df.iloc[0, 1])  # вивід першого рядка, другого стовпця
