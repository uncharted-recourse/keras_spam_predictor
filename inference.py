import argparse
import os
import pickle
import sys
import numpy as np
import pandas as pd

import keras.backend as K
from keras.models import load_model, Model
from keras.preprocessing.sequence import pad_sequences
from attention import AttLayer

class DummyAttLayer(AttLayer):
    def __init__(self, attention_dim=64, **kwargs):
        super().__init__(attention_dim=attention_dim)

class TextHandler(object):
    def __init__(self, tokenizer_path:str, maxlen:int=200, max_sents=None):
        self.tokenizer_path = tokenizer_path
        self._load_tokenizer()
        self.maxlen = maxlen
        self.max_sents = max_sents
        self.id2word = {v: k for k, v in self.tokenizer.word_index.items()}
        self.unk = self.tokenizer.oov_token

    def _load_tokenizer(self):
        with open(self.tokenizer_path, 'rb') as infile:
            tokenizer = pickle.load(infile)

        print('Tokenizer loaded...')
        self.tokenizer = tokenizer

    def _pad_sentence(self, sentence:list, max_num_words:int):
        word_diff = max_num_words - len(sentence)
        padding_words = [0] * word_diff
        sentence.extend(padding_words)
        return sentence
    
    def get_seqs(self, input_sents:list):
        w_id_list = self.tokenizer.texts_to_sequences(input_sents)
        padded_seqs = pad_sequences(w_id_list, value=0, maxlen=self.maxlen, padding='post')
        return padded_seqs, w_id_list

    def reverse_tokenize(self, word_ids:list):
        words = [self.id2word[w_id] if w_id in self.id2word.keys() else self.unk for w_id in word_ids]
        return words

class ModelPredictor(object):
    def __init__(self, model_path:str, tokenizer_path:str, return_prob:bool=True):
        self.model_path = model_path
        self.model = load_model(self.model_path, custom_objects={'AttLayer': DummyAttLayer})
        self.text_handler = TextHandler(tokenizer_path=tokenizer_path)
        self.return_prob = return_prob

    def predict_on_doc(self, input_doc:str):
        x_input, word_ids = self.text_handler.get_seqs([input_doc])
        print(type(x_input))
        print(x_input.shape)
        print(self.text_handler.reverse_tokenize(word_ids[0]))
        y_prob = self.model.predict(x_input)[0][0]
        y_class = 1 if y_prob > 0.5 else 0
        if self.return_prob:
            return y_class, y_prob

        return y_class

    def run_predictions(self, input_path:str, output_path:str=None):
        pred_map = {0: 'friend', 1: 'foe'}
        fnames, result = [], []
        for fname in os.listdir(input_path):
            sys.stdout.write('\r Predicting on doc {}...'.format(fname))
            with open(os.path.join(input_path, fname), encoding='utf8', mode='r') as infile:
                email_doc = infile.read().replace('\t', ' ').replace('\n', ' ')
                model_output = self.predict_on_doc(email_doc)
                result.append(model_output)
                fnames.append(fname)

        out_df = pd.DataFrame()
        if len(result[0]) == 2:
            out_df['predicted_class'] = [pred_map[i[0]] for i in result]
            out_df['predicted_prob'] = [i[-1] for i in result]
        else:
            out_df['predicted_class'] = [pred_map[i] for i in result]

        out_df.index = fnames
        
        if output_path is not None:
            out_df.to_csv(output_path, sep='\t', encoding='utf8')
            print('Output saved!')

        return out_df.head()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=False, default='models/att_rnn_model.h5')
    parser.add_argument('--tokenizer_path', type=str, required=False, default='tokenizer.pkl')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--prediction_save_path', type=str, required=False, default='preds.csv')
    args = parser.parse_args()

    model_predictor = ModelPredictor(model_path=args.model_path,
                                     tokenizer_path=args.tokenizer_path,
                                     return_prob=True)

    model_predictor.run_predictions(input_path=args.data_dir, output_path=args.prediction_save_path)
    print('DONE')
