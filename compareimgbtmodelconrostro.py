import cv2
import numpy as np
import joblib

# Cargar el modelo entrenado
clf = joblib.load('./model/modelo_entrenado.pkl')

# Cargar el codificador de etiquetas
label_encoder = joblib.load('./model/label_encoder.pkl')

# Cargar la imagen a comparar
image_path = './imgPruba/WhatsApp Image 2023-05-25 at 3.28.56 PM.jpeg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Verificar que la imagen se haya cargado correctamente
if image is not None:
    # Redimensionar la imagen si es necesario
    image = cv2.resize(image, (1000, 600))

    # Aplanar la imagen en un formato 1D
    image = image.reshape(1, -1)

    # Obtener las clases del modelo
    classes = clf.classes_

    # Realizar la predicción con el modelo
    predicted_decision = clf.decision_function(image)
    predicted_proba = (1 / (1 + np.exp(-predicted_decision))).flatten()

    # Obtener la etiqueta de la clase predicha
    predicted_label = np.argmax(predicted_proba)
    predicted_person = label_encoder.inverse_transform([predicted_label])

    # Obtener la probabilidad de certeza de la clase predicha
    certainty = predicted_proba[predicted_label]

    # Mostrar el resultado de la predicción
    print("La imagen es reconocida como:", predicted_person[0])
    print("Probabilidad de certeza: {:.2f}%".format(certainty * 100))
else:
    print("No se pudo cargar la imagen correctamente.")
