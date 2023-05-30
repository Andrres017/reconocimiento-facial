import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Directorio raíz de las imágenes de entrenamiento
train_dir = './image'

# Leer las imágenes y etiquetas de entrenamiento
X_train = []
y_train = []

for person_folder in os.listdir(train_dir):
    person_folder_path = os.path.join(train_dir, person_folder)
    if os.path.isdir(person_folder_path):
        for file_name in os.listdir(person_folder_path):
            if file_name.endswith('.jpg'):
                image_path = os.path.join(person_folder_path, file_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    X_train.append(image)
                    y_train.append(int(person_folder))

# Convertir las listas a matrices numpy
X_train = np.array(X_train)
y_train = np.array(y_train)

# Redimensionar las imágenes si es necesario
X_train = np.array([cv2.resize(image, (1000, 600)) for image in X_train])

# Aplanar las imágenes en un formato 1D
X_train = X_train.reshape(X_train.shape[0], -1)

# Codificar las etiquetas de las clases
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)

# Crear el modelo SVC y entrenarlo
clf = SVC(probability=True)
clf.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = clf.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo: {:.2f}%".format(accuracy * 100))

# Guardar el modelo entrenado
model_path = './model/modelo_entrenado.pkl'
joblib.dump(clf, model_path)

print("Modelo entrenado guardado en:", model_path)
