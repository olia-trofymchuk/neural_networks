"""
Завдання:

Здійснити моделювання згенерованого набору даних та візуалізацію результатів.
"""
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Завантаження набору даних з файлу
df = pd.read_csv('regression_data.csv')

# Розділення даних на тренувальний та тестовий набори
train_size = int(0.8 * len(df))
train_df = df[:train_size]
test_df = df[train_size:]

# Моделювання
model = LinearRegression()
model.fit(train_df.drop('target', axis=1), train_df['target'])

# Візуалізація результатів
plt.figure(figsize=(10, 6))
sns.scatterplot(x=test_df.index, y=test_df['target'], color='blue', alpha=0.5)
sns.lineplot(x=test_df.index, y=model.predict(test_df.drop('target', axis=1)))
plt.xlabel('Sample index')
plt.ylabel('Target value')
plt.title('Linear Regression results')
plt.show()
