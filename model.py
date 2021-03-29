import pathlib
import pickle

import torch.nn.functional as F
import numpy as np
import re
import math
from torchtext import data
from torch.autograd import Variable
import torch.nn as nn
import torch

pathlib.PosixPath = pathlib.WindowsPath


class tokenize(object):

    def __init__(self, lang):
        pass

    def tokenizer(self, sentence):
        sentence = re.sub(
            r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]


def k_best_outputs(outputs, out, log_scores, i, k):

    probs, ix = out[:, -1].data.topk(k)
    log_probs = torch.Tensor(
        [math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0, 1)
    k_probs, k_ix = log_probs.view(-1).topk(k)

    row = k_ix // k
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)

    return outputs, log_scores


def nopeak_mask(size, device):
    """Tạo mask được sử dụng trong decoder để lúc dự đoán trong quá trình huấn luyện
     mô hình không nhìn thấy được các từ ở tương lai
    """
    np_mask = np.triu(np.ones((1, size, size)),
                      k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    np_mask = np_mask.to(device)

    return np_mask


def init_vars(src, model, TRG, device, k, max_len):
    """ Tính toán các ma trận cần thiết trong quá trình translation sau khi mô hình học xong
    """
    init_tok = TRG.vocab.stoi['<sos>']

    # tính sẵn output của encoder
    src = src.unsqueeze(0)
    ln = model.linear(src)

    src_mask = (torch.ones(ln.shape[0], 1, ln.shape[1]) == 1).cuda()

    e_output = model.encoder(ln, src_mask)

    outputs = torch.LongTensor([[init_tok]])

    outputs = outputs.to(device)

    trg_mask = nopeak_mask(1, device)
    # dự đoán kí tự đầu tiên
    out = model.out(model.decoder(outputs,
                                  e_output, src_mask, trg_mask))
    out = F.softmax(out, dim=-1)

    probs, ix = out[:, -1].data.topk(k)
    log_scores = torch.Tensor([math.log(prob)
                               for prob in probs.data[0]]).unsqueeze(0)

    outputs = torch.zeros(k, max_len).long()
    outputs = outputs.to(device)
    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0]

    e_outputs = torch.zeros(k, e_output.size(-2), e_output.size(-1))

    e_outputs = e_outputs.to(device)
    e_outputs[:, :] = e_output[0]

    return outputs, e_outputs, log_scores


def multiple_replace(dict, text):
    # Create a regular expression  from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)


def beam_search(src, model, TRG, device, k, max_len):

    outputs, e_outputs, log_scores = init_vars(
        src, model, TRG, device, k, max_len)
    eos_tok = TRG.vocab.stoi['<eos>']

    src_mask = (torch.ones(1, 1, src.shape[0]) == 1).cuda()

    ind = None
    for i in range(2, max_len):

        trg_mask = nopeak_mask(i, device)

        out = model.out(model.decoder(outputs[:, :i],
                                      e_outputs, src_mask, trg_mask))

        out = F.softmax(out, dim=-1)

        outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, k)

        # Occurrences of end symbols for all input sentences.
        ones = (outputs == eos_tok).nonzero()
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long).cuda()
        for vec in ones:
            i = vec[0]
            if sentence_lengths[i] == 0:  # First end symbol has not been found yet
                sentence_lengths[i] = vec[1]  # Position of first end symbol

        num_finished_sentences = len([s for s in sentence_lengths if s > 0])

        if num_finished_sentences == k:
            alpha = 0.7
            div = 1/(sentence_lengths.type_as(log_scores)**alpha)
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.data[0]
            break

    if ind is None:

        length = (outputs[0] == eos_tok).nonzero()[0] if len(
            (outputs[0] == eos_tok).nonzero()) > 0 else -1
        return ' '.join([TRG.vocab.itos[tok] for tok in outputs[0][1:length]])

    else:
        length = (outputs[ind] == eos_tok).nonzero()[0]
        return ' '.join([TRG.vocab.itos[tok] for tok in outputs[ind][1:length]])


def translate_sentence(fea, model, TRG, device, k, max_len):
    """Dịch một câu sử dụng beamsearch
    """

    sentence = torch.tensor(fea)

    sentence = beam_search(sentence, model, TRG, device, k, max_len)

    return multiple_replace({' ?': '?', ' !': '!', ' .': '.', '\' ': '\'', ' ,': ','}, sentence)


if __name__ == '__main__':

    opt = {
        'trg_lang': 'en',
        'max_strlen': 160,
        'batchsize': 1500,
        'device': 'cuda',
        'd_model': 512,
        'n_layers': 6,
        'heads': 8,
        'dropout': 0.1,
        'lr': 0.0001,
        'epochs': 2,
        'printevery': 100,
        'k': 5,
    }
    print(torch.__version__)
    with open('cab.pkl', 'rb') as pickle_file:
        TRG = pickle.load(pickle_file)
    model = torch.load('model_test.pt', map_location=torch.device('cpu'))
    fear = np.load('testv1RN40.npy')
    trans_sent = translate_sentence(
        fear, model, TRG, opt['device'], opt['k'], opt['max_strlen'])
    print('predict: ', trans_sent)