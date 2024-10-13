import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC

# Cargar el conjunto de datos Iris
iris = datasets.load_iris()
X = iris.data[:, :2]  # Tomamos solo las dos primeras características (longitud y ancho del sépalo)
y = iris.target

# Convertimos en una tarea binaria: Solo clasificar Setosa (0) y Versicolor (1)
X = X[y != 2]
y = y[y != 2]

# Crear y entrenar el modelo SVM
model = SVC(kernel='linear', C=1.0)
model.fit(X, y)

# Nueva flor para predecir
nueva_flor = np.array([[5.0, 3.5]])

# Usar el modelo entrenado para hacer una predicción
prediccion = model.predict(nueva_flor)

# Interpretar el resultado
if prediccion == 0:
    print("La nueva flor es Setosa.")
else:
    print("La nueva flor es Versicolor.")

# Función para graficar los resultados y el hiperplano
def plot_svm(X, y, model, nueva_flor):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')

    # Dibujar el hiperplano
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    w = model.coef_[0]
    b = model.intercept_[0]
    x_hyperplane = np.linspace(xlim[0], xlim[1], 100)
    y_hyperplane = -(w[0] * x_hyperplane + b) / w[1]
    plt.plot(x_hyperplane, y_hyperplane, 'k--', label='Hiperplano')

    # Añadir la nueva flor en el gráfico con un color verde
    plt.scatter(nueva_flor[0, 0], nueva_flor[0, 1], color='green', marker='o', s=100, edgecolors='k', label='Nueva Flor')

    # Añadir etiquetas para las clases
    plt.text(x_max - 0.5, y_max - 1, 'Versicolor', fontsize=12, color='red')
    plt.text(x_min + 0.5, y_min + 1, 'Setosa', fontsize=12, color='blue')

    plt.xlabel('Longitud del sépalo (cm)')
    plt.ylabel('Ancho del sépalo (cm)')
    plt.legend()
    plt.title('SVM - Clasificación de Flores (Setosa vs. Versicolor)')
    plt.show()

# Visualizar los resultados
plot_svm(X, y, model, nueva_flor)




