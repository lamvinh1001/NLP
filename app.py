from flask import Flask, render_template, request
from cv2 import cv2
from keras.models import load_model
import numpy as np
from keras.applications import ResNet152
from tts import text_to_speech
from fearture import feature_cap
from flask_cors import cross_origin
resnet = ResNet152(include_top=False, weights='imagenet',
                   input_shape=(224, 224, 3), pooling='avg')
# resnet = load_model('resnet.h5')
app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/after', methods=['POST', 'GET'])
@cross_origin()
def after():
    global resnet
    rq = request.files['file']
    rq.save('static/test.mp4')
    fe = feature_cap(resnet, 'static/test.mp4')
    # data = 'cho trieu'
    gender = request.form['voices']
    text_to_speech('hi my name trieu  stupid', gender)
    return render_template('after.html', data=fe)


if __name__ == "__main__":
    app.run(debug=True, port=8000)
