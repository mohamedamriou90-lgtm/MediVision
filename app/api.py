"""
MEDIVISION - API Server
"""

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('models/detection/medivision_trained.h5')
print("✅ Model loaded successfully!")

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'name': 'MediVision AI',
        'status': 'ready',
        'version': '1.0'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        if 'image' in request.files:
            file = request.files['image']
            img = Image.open(file.stream)
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Preprocess
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = model.predict(img_array)[0][0]
        
        return jsonify({
            'success': True,
            'probability': float(prediction),
            'has_disease': bool(prediction > 0.5),
            'confidence': float(max(prediction, 1-prediction))
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)