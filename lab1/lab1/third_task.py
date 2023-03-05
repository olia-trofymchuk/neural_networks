import csv

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def write_to_csv(filename, data):
    with open(filename, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        for row in data:
            writer.writerow(row)


if __name__ == "__main__":
    # діапазон значень по осі X
    x = np.linspace(0, 10, 100)

    # ідеальні значення функції
    y_true = 2 * x + 1

    # згенеруємо помилки вимірювання
    noise = np.random.normal(scale=0.5, size=x.shape)
    y_measured = y_true + noise

    # побудуємо графік ідеальної функції
    plt.plot(x, y_true, label="True function")

    # побудуємо графік функції з урахуванням помилок вимірювань
    plt.errorbar(x, y_measured, yerr=0.5, fmt="o", label="Measured function")

    # додамо підписи для осей та заголовок
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Measured and true functions")

    # додамо легенду
    plt.legend()

    # показати графік
    plt.show()

    mae = mean_absolute_error(
        y_true, y_measured
    )  # похибка з використанням метрики MAE
    mse = mean_squared_error(
        y_true, y_measured
    )  # похибка з використанням метрики MSE

    # об'єднати дані у список
    rows = []
    rows.append(["X", "Y", "Y_hat", "mAE", "mSE"])
    for i in range(len(x)):
        rows.append([x[i], y_true[i], y_measured[i], mae, mse])

    # записати дані у файл
    write_to_csv("third_task.csv", rows)
