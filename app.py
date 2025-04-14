from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('model.h5')

# Kelas-kelas yang sudah didefinisikan
class_names = ['beterai', 'biologis', 'kaca', 'kardus', 'kertas', 'logam', 'pakaian', 'plastik', 'sampah', 'sepatu']

# Menentukan kategori untuk setiap kelas
organik_classes = ['biologis', 'sampah']
non_organik_classes = ['kaca', 'kardus', 'kertas', 'logam', 'plastik', 'pakaian','sepatu']
berbahaya_classes = ['beterai']

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((100, 100))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# Endpoint prediksi
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    image = preprocess_image(file.read())
    
    prediction = model.predict(image)
    predicted_class_idx = np.argmax(prediction)
    predicted_class = class_names[predicted_class_idx]
    confidence = float(np.max(prediction) * 100)

    if predicted_class in organik_classes:
        category = 'Organik'
    elif predicted_class in non_organik_classes:
        category = 'Non-Organik'
    elif predicted_class in berbahaya_classes:
        category = 'Bahan Berbahaya'
    else:
        category = 'Tidak Diketahui'

    return jsonify({
        'prediksi': predicted_class,
        'persen': f"{confidence:.2f}%",
        'kategori': category  # Menambahkan kategori
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
