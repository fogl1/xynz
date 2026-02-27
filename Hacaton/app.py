# app.py
from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from predict import predict_mask, calculate_area
from tensorflow.keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('best_model.h5')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    mask = predict_mask(model, filepath)
    area_pixels = calculate_area(mask)
    # Можно добавить масштаб, если пользователь его передаёт
    return jsonify({'area_pixels': int(area_pixels)})


if __name__ == '__main__':
    app.run(debug=True)