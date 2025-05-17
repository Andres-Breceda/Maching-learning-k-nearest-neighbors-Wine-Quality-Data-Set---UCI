import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 

data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/refs/heads/main/winequality-red.csv", sep= ";")

data.columns

data.head()

data.describe()

data.info()

data.shape
rangos = data.max() - data.min()
print(rangos)

data.hist(figsize=(10, 6))  # Cambia el tamaño como desees
plt.tight_layout()
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

pair = sns.pairplot(data, hue="quality")
pair._legend.set_title("Quality")  # Asegura que el título de la leyenda diga "Quality"
plt.show()

sns.countplot(x="quality", data =data)

import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

# Gráfico de coordenadas paralelas
plt.figure(figsize=(12, 6))
parallel_coordinates(data, "quality", color=("#E58139", "#39E581", "#8139E5"))

# Rotar etiquetas de los ejes
plt.xticks(rotation=45)  # Cambia a 90 si quieres más inclinación

# Mostrar gráfico
plt.title("Coordenadas paralelas por clase de quality")
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Seleccionar columnas numéricas
data1 = data.select_dtypes(include=[np.number])

# Eliminar 'quality' si está entre las columnas dependientes
cols_to_plot = [col for col in data1.columns if col != 'quality']

# Definir tamaño de la figura y la cuadrícula
n_cols = 4
n_rows = int(np.ceil(len(cols_to_plot) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))

# Aplanar ejes para poder indexarlos fácilmente
axes = axes.flatten()

# Generar los gráficos
for i, col in enumerate(cols_to_plot):
    sns.barplot(x='quality', y=col, data=data, ax=axes[i])
    axes[i].set_title(f'{col} vs quality')

# Eliminar subplots vacíos si hay
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x='quality', y='alcohol', data=data)
plt.title('Distribución de Alcohol por Nivel de Calidad (Quality)')
plt.xlabel('Calidad del Vino')
plt.ylabel('Porcentaje de Alcohol')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x='quality', y='citric acid', data=data)
plt.title('Distribución de Ácido cítrico natural, por Nivel de Calidad (Quality)')
plt.xlabel('Calidad del Vino')
plt.ylabel('Porcentaje de Acido citrico')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x='quality', y='volatile acidity', data=data)
plt.title('Distribución de Alcohol por Nivel de Calidad (Quality)')
plt.xlabel('Calidad del Vino')
plt.ylabel('volatile acidity')
plt.tight_layout()
plt.show()


plt.figure(figsize=(20,20))
sns.heatmap(data.corr(), color = "k", annot=True)

def categorizar_calidad(q):
    if q <= 5:
        return 0  #  calidad Baja
    elif q == 6:
        return 1  # Calidad media
    else:
        return 2  #  calidad Alta


data["label"] = data["quality"].apply(categorizar_calidad)
print(data["label"].value_counts())


data1 = data.copy()
data2 = data.copy()
data.head()

from sklearn.model_selection import train_test_split
X = data.drop( columns=["label","quality"])
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify= y, test_size = 0.2, random_state = 42)

import matplotlib.pyplot as plt
import seaborn as sns

# Crear figura y ejes (2 filas x 3 columnas)
fig, axis = plt.subplots(2, 3, figsize=(15, 7))

# Elegir una paleta de colores para más clases de calidad (si hay más de 3)
palette = sns.color_palette("tab10", as_cmap=False, n_colors=data["quality"].nunique())

# Graficar combinaciones clave usando quality como hue
sns.scatterplot(ax=axis[0, 0], data=data, x="alcohol", y="sulphates", hue="quality", palette=palette)
sns.scatterplot(ax=axis[0, 1], data=data, x="alcohol", y="volatile acidity", hue="quality", palette=palette)
sns.scatterplot(ax=axis[0, 2], data=data, x="alcohol", y="citric acid", hue="quality", palette=palette)
sns.scatterplot(ax=axis[1, 0], data=data, x="sulphates", y="volatile acidity", hue="quality", palette=palette)
sns.scatterplot(ax=axis[1, 1], data=data, x="sulphates", y="citric acid", hue="quality", palette=palette)
sns.scatterplot(ax=axis[1, 2], data=data, x="citric acid", y="volatile acidity", hue="quality", palette=palette)

# Ajustar distribución y mostrar
plt.tight_layout()
plt.show()

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors= 1)
model.fit(X_train,y_train)

y_predict = model.predict(X_test)
y_predict

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

accuracy_score(y_test, y_predict)

confusion_matrix(y_test,y_predict)

print(classification_report(y_test, y_predict))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

accuracies = []

# Probar k de 1 a 20
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# Graficar accuracy vs k
plt.figure(figsize=(8, 5))
plt.plot(range(1, 21), accuracies, marker='o')
plt.title("Accuracy vs número de vecinos (k)")
plt.xlabel("Número de vecinos (k)")
plt.ylabel("Accuracy")
plt.xticks(range(1, 21))
plt.grid(True)
plt.show()

from sklearn.model_selection import train_test_split
X = data1.drop( columns=["label","quality"])
y = data1["label"]

x_train, x_test, y_train, y_test = train_test_split(X, y, stratify= y, test_size = 0.2, random_state = 42)


# 3. Aplicar SMOTE solo al entrenamiento
from imblearn.over_sampling import SMOTE
smote = SMOTE(k_neighbors=5, random_state=42)
x_train, y_train = smote.fit_resample(x_train, y_train)

# 4. Escalar después del resampleo
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)  # ¡No uses fit en test!

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

accuracies = []

for k in range(1, 21):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)


# 6. Graficar Accuracy vs K
plt.figure(figsize=(8, 5))
plt.plot(range(1, 21), accuracies, marker='o')
plt.title("Accuracy vs número de vecinos (k)")
plt.xlabel("Número de vecinos (k)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.xticks(range(1, 21))
plt.show()


# 7. Mostrar matriz de confusión y reporte con el mejor k
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

best_k = np.argmax(accuracies) + 1
print(f"Mejor k: {best_k}")

model_final = KNeighborsClassifier(n_neighbors=best_k)
model_final.fit(x_train_scaled, y_train)
y_final_pred = model_final.predict(x_test_scaled)

print(confusion_matrix(y_test, y_final_pred))
print(classification_report(y_test, y_final_pred))

model = KNeighborsClassifier(n_neighbors=2)
model.fit(x_train_scaled, y_train)
y_pred = model.predict(x_test_scaled)
acc = accuracy_score(y_test, y_pred)
acc