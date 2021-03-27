from flask import Flask, render_template, request
from cv2 import cv2
from keras.models import load_model
import numpy as np
from keras.applications import ResNet152
import pickle
from fearture import feature_cap
resnet = ResNet152(include_top=False, weights='imagenet',
                   input_shape=(224, 224, 3), pooling='avg')
# resnet = load_model('resnet.h5')

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/after', methods=['GET', 'POST'])
def after():
    global resnet
    rq = request.files['file1']
    rq.save('static/test1.mp4')
    rq = str(rq).split(' ')
    rq = rq[1].replace("'", '')
    url_video = 'static/' + rq

    fe = feature_cap(resnet, url_video)
    return render_template('after.html', data=fe)


if __name__ == "__main__":
    app.run(debug=True, port=8000)
