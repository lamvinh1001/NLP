import torch
from flask import Flask, render_template, request
from models import predict_cap
from keras.applications import ResNet152
from fearture import feature_cap
from flask_cors import cross_origin
from googletts import speak
import re
import spacy
import gc
gc.collect()
torch.cuda.empty_cache()
resnet = ResNet152(include_top=False, weights='imagenet',
                   input_shape=(224, 224, 3), pooling='avg')

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


class tokenize(object):

    def __init__(self, lang):
        self.nlp = spacy.load(lang)

    def tokenizer(self, sentence):
        sentence = re.sub(
            r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/after', methods=['POST', 'GET'])
@cross_origin()
def after():
    global resnet
    rq = request.files['file']
    cap = ''
    file_path = 'static/test.mp4'
    rq.save(file_path)
    fears = feature_cap(resnet, file_path)

    cap += predict_cap(fears)
    speak(cap)
    return render_template('after.html', data=cap)


if __name__ == "__main__":
    app.run(debug=True)
