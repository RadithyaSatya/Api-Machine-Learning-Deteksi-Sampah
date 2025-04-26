from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import os
from flask_cors import CORS  

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

CORS(app)  
model = tf.keras.models.load_model('model.h5')

class_names = ['beterai', 'biologis', 'kaca', 'kardus', 'kertas', 'logam', 'pakaian', 'plastik', 'sampah', 'sepatu']
organik_classes = ['biologis', 'sampah', 'kertas']
non_organik_classes = ['kaca', 'kardus', 'logam', 'plastik', 'pakaian', 'sepatu'] 
berbahaya_classes = ['beterai']

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((100, 100))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

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
        'kategori': category  
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
