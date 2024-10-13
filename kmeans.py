import pandas as pd
import numpy as np
import seaborn as sns  # Asegúrate de importar seaborn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Cargar el archivo Excel
data = pd.read_excel('Data10.xlsx')

# Seleccionar características
X_kmeans = data[['Precio actual', 'Precio final']]

# Encontrar el número óptimo de clústeres utilizando el método del codo
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_kmeans)
    inertia.append(kmeans.inertia_)

# Graficar el método del codo
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Método del Codo para K-means')
plt.xlabel('Número de Clústeres')
plt.ylabel('Inercia')
plt.show()

# Elegir un número de clústeres y ajustar el modelo
n_clusters = 3  # Por ejemplo, puedes elegir 3 clústeres
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_kmeans)

# Graficar los clústeres
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='Precio actual', y='Precio final', hue='Cluster', palette='viridis')
plt.title('Clustering de K-means')
plt.xlabel('Precio Actual')
plt.ylabel('Precio Final')
plt.legend(title='Clúster')
plt.show()


