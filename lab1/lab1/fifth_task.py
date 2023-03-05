import numpy as np
import pandas as pd
import tensorflow as tf

# Створення тензора розміром 2x3x4
a = tf.constant(
    [
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
        [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
    ]
)

# Створення змінного тензора
b = tf.Variable([4, 5, 6])

# Оновлення значення
b.assign([7, 8, 9])

# Видалення тензора
del b

# Зміна форми тензора з розміром 2x3x4 на 3x8
new_shape = (3, 8)
reshaped_tensor = tf.reshape(a, new_shape)

# Виведення значень тензорів
tf.print(a, "\n")
tf.print(reshaped_tensor)

# Створення масиву NumPy
np_array = np.array([[1, 2], [3, 4]])

# Перетворення масиву NumPy у тензор Tensorflow
tf_tensor = tf.convert_to_tensor(np_array)

# Виведення значень масиву та тензора
print(f"NumPy array:\n {np_array}")
print(f"Tensorflow tensor:\n {tf_tensor}")

# Створення DataFrame Pandas
data = {"A": [1, 2, 3], "B": [4, 5, 6]}
df = pd.DataFrame(data)

# Перетворення DataFrame Pandas у тензор Tensorflow
tf_tensor = tf.convert_to_tensor(df.values)

# Виведення значень DataFrame та тензора
print(f"Pandas DataFrame:\n {df}")
print(f"Tensorflow tensor:\n {tf_tensor}")
