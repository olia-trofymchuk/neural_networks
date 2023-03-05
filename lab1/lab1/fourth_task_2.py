from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# збираємо та підготовлюємо дані для моделювання
iris = (
    load_iris()
)  # набір даних Iris для класифікації типу квітів на основі їх розмірів
x = iris.data
y = iris.target

# розділяємо дані на навчальний та тестовий набори
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# створюємо класифікатор і навчаємо його на навчальному наборі
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

# оцінюємо ефективність моделі на тестовому наборі
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# використовуємо навчений класифікатор для передбачення класів для нових даних
new_data = [[5.1, 3.5, 1.4, 0.2], [6.3, 3.3, 4.7, 1.6]]
new_predictions = clf.predict(new_data)
print("New predictions:", new_predictions)
