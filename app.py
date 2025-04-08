from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import math
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, resources={
    r"/predict": {"origins": "*"},
    r"/": {"origins": "*"}
})

detector = HandDetector(maxHands=1)

# Fix for DepthwiseConv2D custom layer
class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(**kwargs)

# Load the model
model = tf.keras.models.load_model(
    "Model/keras_model.h5",
    custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D},
    compile=False
)

# Load labels
with open("Model/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        img_bytes = file.read()
        if not img_bytes:
            return jsonify({'error': 'Empty image file'}), 400

        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400

        hands, _ = detector.findHands(img)
        if not hands:
            return jsonify({'result': 'No hand detected', 'confidence': 0})

        hand = hands[0]
        x, y, w, h = hand['bbox']
        offset = 20
        imgSize = 300
        
        # Ensure the crop coordinates are within image bounds
        y_start = max(0, y - offset)
        y_end = min(img.shape[0], y + h + offset)
        x_start = max(0, x - offset)
        x_end = min(img.shape[1], x + w + offset)
        
        imgCrop = img[y_start:y_end, x_start:x_end]
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + hCal, :] = imgResize

        imgInput = cv2.resize(imgWhite, (224, 224))
        imgInput = np.expand_dims(imgInput, axis=0) / 255.0

        predictions = model.predict(imgInput)
        index = np.argmax(predictions)
        confidence = float(predictions[0][index])

        if confidence > 0.7:
            return jsonify({
                'result': labels[index],
                'confidence': confidence
            })
        else:
            return jsonify({
                'result': 'Uncertain',
                'confidence': confidence
            })

    except Exception as e:
        app.logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)