import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.optimizers import Adamax
from PIL import Image  # Import the Image module from PIL

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def welcome():
    return "Welcome -- It's our Graduation AI Model"
@app.route('/', methods=['POST'])
def generateImage():
    if 'image_data' not in request.files:
        return jsonify({'error': 'No image file provided'})

    image_file = request.files['image_data']


    # Load the pre-trained model and compile it
    loaded_model = tf.keras.models.load_model('Pneumonia_Detection_EfficientNet.h5', compile=False)
    
    loaded_model.compile(optimizer=Adamax(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Open and preprocess the image
    image = Image.open(image_file)

    img = image.resize((224, 224))

    img_array = tf.keras.preprocessing.image.img_to_array(img)

    img_array = tf.expand_dims(img_array, 0)

    # Make predictions
    predictions = loaded_model.predict(img_array)

    class_labels = ['Normal', 'Pneumonia']

    score = tf.nn.softmax(predictions[0])

    predicted_label = class_labels[tf.argmax(score)]

    response_data = {'Predicted class': predicted_label}

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0')