import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Cargar el archivo Excel
data = pd.read_excel('Data10.xlsx')

# Preprocesar los datos
# Convertir 'Estado' a variable numérica
data['Estado'] = data['Estado'].map({'Alto': 1, 'Bajo': 0})

# Seleccionar las características y la etiqueta
X = data[['Precio actual', 'Precio final']]  # Seleccionamos las características
y = data['Estado']  # Seleccionamos la etiqueta

# Dividir el conjunto de datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear y entrenar el modelo SVM
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
print("Matriz de Confusión:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("\nInforme de Clasificación:")
print(classification_report(y_test, y_pred))

# Graficar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bajo', 'Alto'], yticklabels=['Bajo', 'Alto'])
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.title('Matriz de Confusión')
plt.show()

# Obtener métricas de clasificación
precision_0 = 0.55  # Sustituye con el valor real
recall_0 = 0.02     # Sustituye con el valor real
f1_0 = 0.04         # Sustituye con el valor real

precision_1 = 0.52  # Sustituye con el valor real
recall_1 = 0.98     # Sustituye con el valor real
f1_1 = 0.68         # Sustituye con el valor real

# Graficar las métricas de clasificación
metricas = ['Precisión', 'Recall', 'F1-score']
valores_clase_0 = [precision_0, recall_0, f1_0]
valores_clase_1 = [precision_1, recall_1, f1_1]

x = np.arange(len(metricas))  # la ubicación de las etiquetas
width = 0.35  # el ancho de las barras

fig, ax = plt.subplots(figsize=(8, 6))
bars1 = ax.bar(x - width/2, valores_clase_0, width, label='Clase 0 (Bajo)')
bars2 = ax.bar(x + width/2, valores_clase_1, width, label='Clase 1 (Alto)')

# Añadir etiquetas y título
ax.set_ylabel('Valores')
ax.set_title('Métricas de Clasificación por Clase')
ax.set_xticks(x)
ax.set_xticklabels(metricas)
ax.legend()

# Mostrar el gráfico
plt.ylim(0, 1)  # Limitar el eje y entre 0 y 1
plt.show()

# Función para predecir el estado de un nuevo registro
def predecir_estado(precio_actual, precio_final):
    nuevo_dato = np.array([[precio_actual, precio_final]])
    nuevo_dato = scaler.transform(nuevo_dato)  # Escalar el nuevo dato
    prediccion = model.predict(nuevo_dato)
    return "Alto" if prediccion[0] == 1 else "Bajo"

# Ejemplo de uso de la función de predicción
estado_nuevo = predecir_estado(20.0, 15.0)
print(f"El estado de la nueva entrada es: {estado_nuevo}")


