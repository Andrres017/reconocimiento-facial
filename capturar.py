import cv2
import os

# Directorio de videos
videos_dir = './videos'

# Directorio de salida para las imágenes
output_dir = './image'
os.makedirs(output_dir, exist_ok=True)

# Número máximo de imágenes a extraer por video
num_images_per_video = 200

# Cargar el clasificador pre-entrenado de detección de rostros
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Recorrer todos los archivos en el directorio de videos
for filename in os.listdir(videos_dir):
    video_path = os.path.join(videos_dir, filename)

    # Cargar el video
    cap = cv2.VideoCapture(video_path)

    # Obtener información del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Calcular la frecuencia de muestreo para extraer las imágenes
    sampling_rate = int(total_frames / num_images_per_video)

    # Crear un directorio específico para este video
    video_output_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
    os.makedirs(video_output_dir, exist_ok=True)

    # Inicializar contador de imágenes
    image_count = 0
    frame_count = 0

    while True:
        # Leer el siguiente cuadro del video
        ret, frame = cap.read()

        # Verificar si se pudo leer el cuadro
        if not ret:
            break

        # Incrementar el contador de cuadros
        frame_count += 1

        # Verificar si se debe extraer una imagen en este cuadro
        if frame_count % sampling_rate == 0:
            # Convertir el cuadro a escala de grises
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detectar los rostros en el cuadro
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Verificar si se encontraron rostros en el cuadro
            if len(faces) > 0:
                # Tomar el primer rostro detectado
                (x, y, w, h) = faces[0]

                # Recortar la región del rostro
                face = frame[y:y+h, x:x+w]

                # Redimensionar la región del rostro a un tamaño fijo
                face = cv2.resize(face, (200, 200))

                # Generar el nombre de archivo de la imagen
                image_name = f'image_{image_count:03d}.jpg'
                image_path = os.path.join(video_output_dir, image_name)

                # Guardar la imagen en disco
                cv2.imwrite(image_path, face)
                print(f'Imagen guardada: {image_name}')

                # Incrementar el contador de imágenes
                image_count += 1

                # Verificar si se alcanzó el número máximo de imágenes por video
                if image_count >= num_images_per_video:
                    break

    # Liberar los recursos y cerrar el video
    cap.release()

print('Proceso completado')
