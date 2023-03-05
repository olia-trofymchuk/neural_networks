from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd

# завантаження даних
data = pd.read_csv("second_data.csv")

# розділення на тренувальний та тестовий набори
x_train, x_test, y_train, y_test = train_test_split(
    data.drop("target", axis=1), data["target"], test_size=0.2
)

# навчання моделі
model = LinearRegression()
model.fit(x_train, y_train)

# прогнозування на тестовому наборі
y_pred = model.predict(x_test)

# оцінка якості моделі за допомогою MAE
mae = mean_absolute_error(y_test, y_pred)
print("MAE: ", mae)
