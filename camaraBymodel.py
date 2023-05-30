import cv2
import numpy as np
import joblib

# Cargar el modelo entrenado
clf = joblib.load('./model/modelo_entrenado.pkl')

# Cargar el codificador de etiquetas
label_encoder = joblib.load('./model/label_encoder.pkl')

# Cargar el modelo de detección de rostros
face_cascade = cv2.CascadeClassifier(
    './xml/haarcascade_frontalface_default.xml')

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while True:
    # Capturar frame por frame
    ret, frame = cap.read()

    # Convertir el frame a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en la imagen
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Verificar si se detectó al menos un rostro
    if len(faces) > 0:
        # Tomar solo la primera detección de rostro (puedes modificar esto según tus necesidades)
        (x, y, w, h) = faces[0]

        # Recortar el área del rostro de la imagen
        face = gray[y:y + h, x:x + w]

        # Redimensionar el rostro si es necesario
        face = cv2.resize(face, (1000, 600))

        # Aplanar el rostro en un formato 1D
        face = face.reshape(1, -1)

        # Realizar la predicción con el modelo
        predicted_proba = clf.predict_proba(face)
        predicted_label = clf.predict(face)

        # Decodificar la etiqueta de la clase predicha
        predicted_person = label_encoder.inverse_transform(predicted_label)

        # Obtener el porcentaje de confianza
        confidence = np.max(predicted_proba) * 100

        # Mostrar el resultado de la predicción en la imagen
        if predicted_person[0] is not None:
            text = "Persona: {} ({:.2f}% confianza)".format(
                predicted_person[0], confidence)
            cv2.putText(frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No identificado", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "No se detectó rostro", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Mostrar la imagen en una ventana
    cv2.imshow('Identificación en tiempo real', frame)

    # Salir del bucle al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
