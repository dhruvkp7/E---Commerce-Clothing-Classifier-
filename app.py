from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model('ecommerce_classifier_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    img_file = request.files['file']
    if not img_file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
        return jsonify({'error': 'Invalid file type. Please upload a PNG, JPG, or JPEG image'}), 400
    
    img = image.load_img(img_file, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    class_label = {v: k for k, v in train_generator.class_indices.items()}
    response = {'predicted_class': class_label[class_idx], 'confidence': float(predictions[0][class_idx])}

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
