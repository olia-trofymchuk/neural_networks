import random

import numpy as np

"""
Скаляр - один елемент масиву
"""
x = np.array(5)  # create
print(f"Element: {x}")  # read
x = 20  # update
print(f"Element after updating: {x}\n")
del x  # delete

"""
Вектор - масив з одномірними даними
"""
vector = np.arange(0, 10)  # create

print(f"All array: {vector}")  # read all array
print(f"First element: {vector[0]}")  # read just first element

vector[0] = 1  # update first element

v = np.delete(vector, 3)  # delete element 3
print(f"Vector after deleting element: {vector}\n")

"""
Матриця - масив з двомірними даними
"""
matrix = np.arange(1, 10).reshape(3, 3)  # create

print(f"Matrix:\n {matrix}")  # read all array
print(
    f"Second element in first line: {matrix[0][1]}\n"
)  # read second element in first line

matrix[0][1] = 5  # update second element in first line

matrix = np.delete(matrix, 1, axis=0)  # delete second line
print(f"Matrix after deleting second line and updating element:\n {matrix}")
