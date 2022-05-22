import cv2
import numpy as np
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from PIL import Image

IMG_WIDTH = 350
IMG_HEIGHT = 350
labels = ['Abyssinian',
 'Bengal',
 'Birman',
 'Bombay',
 'British',
 'Egyptian',
 'Maine',
 'Persian',
 'Ragdoll',
 'Russian',
 'Siamese',
 'Sphynx',
 'american',
 'basset',
 'beagle',
 'boxer',
 'chihuahua',
 'english',
 'german',
 'great',
 'havanese',
 'japanese',
 'keeshond',
 'leonberger',
 'miniature',
 'newfoundland',
 'pomeranian',
 'pug',
 'saint',
 'samoyed',
 'scottish',
 'shiba',
 'staffordshire',
 'wheaten',
 'yorkshire']
model = load_model("FullBreed86.h5")

def recognize_breed(image):
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.
    yp_class = model.predict(np.array([image]))
    animal_breed = labels[yp_class.argmax()]
    print(yp_class.argmax())
    return animal_breed

app = Flask(__name__)
@app.route("/predict", methods=["POST"])
def predict():
    
    image = request.files['image']
    image = Image.open(image)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    label = recognize_breed(image)

    return jsonify({"breed": label})


if __name__ == "__main__":
    app.run(debug=True)


# END: how to use as API