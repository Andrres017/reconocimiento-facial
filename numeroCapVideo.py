import cv2

# Obtener el nombre del backend de la capturadora de video
backend_name = cv2.VideoCapture(0).getBackendName()

print("Backend de la capturadora de video:", backend_name)
