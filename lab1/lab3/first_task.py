import tensorflow as tf
from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Завантаження набору даних Fashion MNIST відповідно до завдання
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Визначаємо класи згідно із варіантом, які будуть використовуватися для моделювання
classes = [7, 0, 4]

# Фільтруємо дані за вибраними класами
X_train = []
Y_train = []
for i in range(len(y_train)):
    if y_train[i] in classes:
        X_train.append(x_train[i])
        Y_train.append(classes.index(y_train[i]))

X_test = []
Y_test = []
for i in range(len(y_test)):
    if y_test[i] in classes:
        X_test.append(x_test[i])
        Y_test.append(classes.index(y_test[i]))

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

# Нормалізуємо дані
X_train = X_train / 255.0
X_test = X_test / 255.0

# Визначимо модель нейронної мережі
model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=[28, 28]),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ]
)

# Встановимо параметри моделі
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"],
)

# Навчаємо модель
history = model.fit(X_train, Y_train, epochs=10, validation_split=0.2)

# Оцінюємо точність моделі на тестових даних
test_loss, test_accuracy = model.evaluate(X_test, Y_test)

# Виводимо точність моделі на тестових даних
print("Test Accuracy:", test_accuracy)

# Будуємо Confusion Matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
conf_matrix = tf.math.confusion_matrix(labels=Y_test, predictions=y_pred_classes)

# Візуалізуємо Confusion Matrix
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="g")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
