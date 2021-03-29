import re
import pickle
from torchtext import data
import spacy
import pathlib
import torch
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


# class tokenize(object):

#     def __init__(self, lang):
#         self.nlp = spacy.load(lang)

#     def tokenizer(self, sentence):
#         sentence = re.sub(
#             r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
#         sentence = re.sub(r"[ ]+", " ", sentence)
#         sentence = re.sub(r"\!+", "!", sentence)
#         sentence = re.sub(r"\,+", ",", sentence)
#         sentence = re.sub(r"\?+", "?", sentence)
#         sentence = sentence.lower()
#         return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]


with (open('data/cab.pkl', 'rb')) as pickel_file:
    TRG = pickle.load(pickel_file)
print(len(TRG.vocab))
