from sklearn.preprocessing import LabelEncoder
import joblib

# Obtener las etiquetas de los datos de entrenamiento
# Lista de etiquetas correspondientes a tus datos de entrenamiento
labels = [36182145, 1075300651, 1075307011]

# Inicializar y ajustar el codificador de etiquetas
label_encoder = LabelEncoder()
label_encoder.fit(labels)

# Guardar el codificador de etiquetas
joblib.dump(label_encoder, './model/label_encoder.pkl')
