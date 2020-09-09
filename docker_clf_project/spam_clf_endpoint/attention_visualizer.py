import html
import os
import pickle
import numpy as np
import pandas as pd

import keras.backend as K
from nltk.tokenize import sent_tokenize
from keras.models import load_model, Model
from keras.preprocessing.sequence import pad_sequences
from keras.layers.wrappers import TimeDistributed
from attention import AttLayer
from hatt_model import HattKerasModel

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class DummyAttLayer(AttLayer):
    def __init__(self, attention_dim=64, **kwargs):
        super().__init__(attention_dim=attention_dim)

class TextHandler(object):
    def __init__(self, tokenizer, maxlen:int=150, max_sents=None):
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.max_sents = max_sents
        self.id2word = {v: k for k, v in self.tokenizer.word_index.items()}
        self.unk = self.tokenizer.oov_token

    def _pad_sentence(self, sentence:list, max_num_words:int):
        word_diff = max_num_words - len(sentence)
        padding_words = [0] * word_diff
        sentence.extend(padding_words)
        return sentence
    
    def get_seqs(self, input_sents:list):
        w_id_list = self.tokenizer.texts_to_sequences(input_sents)
        padded_seqs = pad_sequences(w_id_list, value=0, maxlen=self.maxlen, padding='post')
        return padded_seqs, w_id_list

    def process_doc(self, input_docs:list):
        # Get input doc(s) and get additional data
        x_batch, num_sents, num_words = [], [], []
        for d in input_docs:
            sents = sent_tokenize(d.strip())
            num_sents.append(len(sents))
            x_sent_toks = self.tokenizer.texts_to_sequences(sents)
            # Trim extra-long sentences
            x_sent_toks = [s[:self.maxlen] for s in x_sent_toks]
            num_words.append(max([len(s) for s in x_sent_toks]))
            x_batch.append(x_sent_toks)
        
        # Data has been batched - now pad and return
        max_num_words = max(num_words)
        if (self.max_sents is None) or (max(num_sents) < self.max_sents):
            max_num_sents = max(num_sents)
        else:
            max_num_sents = self.max_sents

        for doc_idx, doc in enumerate(x_batch):
            # First pad documents with any additional sentences
            if len(doc) < max_num_sents:
                sent_diff = max_num_sents - len(doc)
                padding_sents = [[0] * max_num_words for _ in range(sent_diff)]
                # Add additional sentences
                doc.extend(padding_sents)
                # x_batch[doc_idx] = doc
                for sent_idx, sent in enumerate(doc):
                    if len(sent) < max_num_words:
                        padded_sent = self._pad_sentence(sentence=sent, max_num_words=max_num_words)
                        doc[sent_idx] = padded_sent

                x_batch[doc_idx] = np.asarray(doc)
            else:
                for sent_idx, sent in enumerate(doc):
                    if len(sent) < max_num_words:
                        padded_sent = self._pad_sentence(sentence=sent, max_num_words=max_num_words)
                        doc[sent_idx] = padded_sent
                x_batch[doc_idx] = np.asarray(doc)
        
        # x_batch_out = np.asarray(x_batch)
        x_batch_out = np.asarray(x_batch[:])
        return x_batch_out


    def reverse_tokenize(self, word_ids:list):
        words = [self.id2word[w_id] if w_id in self.id2word.keys() else self.unk for w_id in word_ids]
        return words

class AttentionVisualizer(object):
    def __init__(self, model_path:str, tokenizer):
        self.model_path = model_path
        self.model = load_model(self.model_path, custom_objects={'AttLayer': DummyAttLayer})
        self._get_attention_weights()
        self.text_handler = TextHandler(tokenizer)
        
    def _get_attention_weights(self, att_layer_name:str='dummy_att_layer_1'):
        att_w = self.model.get_layer(att_layer_name).get_weights()
        self.W, self.b, self.u = att_w

    def _get_recurrent_activations(self, x):
        rec_layer_model = Model(inputs=self.model.layers[0].input, outputs=self.model.get_layer('gru_1').output)
        return rec_layer_model.predict(x)

    def _compute_att_weights(self, h, W, b, u):
        first_dot = np.dot(h, W) + b
        uit = np.tanh(first_dot)
        ait = np.dot(uit, u)
        ait = np.squeeze(ait, -1)
        ait = np.exp(ait)
        ait /= np.sum(ait, axis=1, keepdims=True)
        return ait

    def get_attention_scores(self, input_text, return_word_ids:bool=False):
        if isinstance(input_text, str):
            x_input, word_ids = self.text_handler.get_seqs([input_text])
        elif isinstance(input_text, list) or isinstance(input_text, np.ndarray):
            x_input, word_ids = self.text_handler.get_seqs(input_text)
        h = self._get_recurrent_activations(x_input)
        att_w = np.squeeze(self._compute_att_weights(h, self.W, self.b, self.u))

        if return_word_ids:
            return att_w, word_ids
        
        return att_w

    def plot_att_weights(self, sent, weights, maxlen=150):
        sent_len = len(sent.split())
        att_df = pd.DataFrame({'word': sent.split(), 'weight': weights[:sent_len]})
        att_df.set_index('word')[::-1].plot(kind='barh', figsize=(6, 8))
        return

    def pretty_predict(self, text, threshold:float=0.5, hit:bool=False):
        attn, word_ids = self.get_attention_scores(input_text=text, return_word_ids=True)
        toks = self.text_handler.reverse_tokenize(word_ids[0])
        # hit = scores[0][1] > threshold
        # attn_len = len(attn[0])
        result_str = "<div style=\"padding-top: 20px; color: {}\">".format("#ff0000" if hit else "#0000ff")
        tmp = attn - np.min(attn)
        tmp /= np.max(tmp)
        for i, tok in enumerate(toks):
            tok_score = tmp[i] # if i < attn_len else 0.0001
            #tok_score = 1+(3*tok_score if hit else (1-tok_score))
            # tok_score = 1 + (3 * attn[i] if hit else (1 - tok_score))
            if hit:
                tok_score = 1 + (3 * tmp[i])
            else:
                tok_score = 3 * abs(1 - tok_score)
            result_str += "<span style=\"font-size: {}pt\">{}</span><span style=\"font-size:20pt\">&nbsp;</span>".format(8*tok_score, html.escape(tok))

        formatter_score = '> 0.5' if hit else '< 0.5'
        # result_str += "</div><br /><div>Score: {0:.4f}</div>".format(formatter_score)
        result_str += "</div><br /><div>Score: {}</div>".format(formatter_score)

        display(HTML(result_str))
        return attn

# Need this to load the weights for the hierarchical model
class DummyConfig:
    train_file = ''
    valid_file = ''
    vocab_size = 70000
    log_file = None
    embedding_dim = 128
    hidden_dim = 64
    top_hidden_dim = 32
    embedding_dropout_rate = 0.35
    hidden_layers = 1
    recurrent_cell = 'gru'
    opt = 'adam'
    return_attention_weights = False
    n_epochs = 1
    batch_size = 128
    bidirectional = False

class HierarchicalAttentionViz(AttentionVisualizer):
    def __init__(self, model_path, tokenizer):
        self.model_path = model_path
        self.model = self.load_model()
        self._get_attention_weights(att_layer_name='att_layer_2')
        self.text_handler = TextHandler(tokenizer)
        # super().__init__(model_path, tokenizer)
        self._get_doc_rnn()
        self._get_sent_rnn()
        self._get_sent_attention_weights()

    def load_model(self):
        config = DummyConfig()
        model_obj = HattKerasModel(config, mode='predict')
        pretrained_model = model_obj.model
        pretrained_model.load_weights(self.model_path)
        return pretrained_model

    def _get_sent_rnn(self):
        encoder = self.model.get_layer('time_distributed_1')
        sent_encoder = Model(inputs=encoder.layer.layers[0].input,
                             outputs=encoder.layer.layers[2].output)
        sent_encoder = TimeDistributed(sent_encoder)(self.doc_level_model.input)
        sent_level_model = Model(inputs=self.model.input, outputs=sent_encoder)
        self.sent_level_model = sent_level_model
    
    def _get_doc_rnn(self):
        doc_level_model = Model(inputs=self.model.layers[0].input, outputs=self.model.layers[2].output)
        self.doc_level_model = doc_level_model

    def _get_sent_attention_weights(self):
        W, b, u = self.model.get_layer('time_distributed_1').layer.get_layer('att_layer_1').get_weights()
        self.W_sent, self.b_sent, self.u_sent = W, b, u

    def get_sent_attention_scores(self, input_docs:list, return_word_ids:bool=False):
        x_input = self.text_handler.process_doc(input_docs=input_docs)
        h_sent_out = self.sent_level_model.predict(x_input)

        att_w = self._compute_att_weights(h=h_sent_out, W=self.W_sent, b=self.b_sent, u=self.u_sent)
        if return_word_ids:
            word_id_sents = []
            for doc_idx in range(x_input.shape[0]):
                word_ids = []
                for sent in x_input[doc_idx]:
                    sent_word_ids = list(sent)
                    if 0 in sent_word_ids:
                        word_ids.append(sent_word_ids[:sent_word_ids.index(0)])
                    else:
                        word_ids.append(sent_word_ids)
                    word_id_sents.append(word_ids)
            return att_w, word_id_sents

        return att_w

    def get_doc_attention_scores(self, input_docs:list):
        x_input = self.text_handler.process_doc(input_docs=input_docs)
        h_doc_out = self.doc_level_model.predict(x_input)

        att_w = self._compute_att_weights(h=h_doc_out, W=self.W, b=self.b, u=self.u)
        return att_w

    def get_visualization_data(self, input_docs:list):
        class_map = {0: 'HAM', 1: 'SPAM'}

        x_input = self.text_handler.process_doc(input_docs=input_docs) # Probably refactor this so we don't have to do it 3 times
        doc_prediction_prob = self.model.predict(x_input)[0][0]
        doc_prediction = 1 if doc_prediction_prob > 0.5 else 0
        # doc_score = (1 - doc_prediction_prob) if doc_prediction == 0 else doc_prediction_prob
        doc_score = round(float(doc_prediction_prob), 3)

        sent_attention_scores, word_ids = self.get_sent_attention_scores(input_docs, return_word_ids=True)
        doc_attention_scores = self.get_doc_attention_scores(input_docs)

        # Create data map
        model_output = {}
        model_output['doc_prediction'] = class_map[doc_prediction]
        model_output['doc_score'] = doc_score # round(float(doc_score * 100), 2)
        
        sentence_data = []
        for doc_idx in range(len(input_docs)):
            words = [self.text_handler.reverse_tokenize(sent) for sent in word_ids[doc_idx]]
            sentence_container = []
            for sent_idx, sent_score in enumerate(sent_attention_scores[doc_idx]):
                sent_map = {}
                sent_map['words'] = words[sent_idx]
                word_scores = sent_score - np.min(sent_score)
                # May be cases where we get division-by-zero errors
                # Still need to investigate whether this is a larger issue
                if np.max(word_scores) == 0.0:
                    continue
                word_scores /= np.max(word_scores)
                if doc_prediction == 0:
                    word_scores = 1 - word_scores
                word_scores = list(np.squeeze(word_scores))[:len(sent_map['words'])]
                sent_map['word_scores'] = [float(s) for s in word_scores]
                sent_map['sent_score'] = float(doc_attention_scores[doc_idx][sent_idx])
                # sentence_container.append(sent_map)
                sentence_data.append(sent_map)
            # sentence_data.append(sentence_container)

        model_output['sentences'] = sentence_data

        return model_output


# Test compiling / initialization
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    model_path = 'models/hatt_rnn_model-epoch10-loss0.00.h5'
    tok_path = 'hatt_tokenizer.pkl'

    with open(tok_path, mode='rb') as infile:
        tokenizer = pickle.load(infile)

    han_viz = HierarchicalAttentionViz(model_path=model_path, tokenizer=tokenizer)
    print('Hierarchical visualizer initialized!')

    print(dir(han_viz))

    sent = 'Here is an example sentence .'
    print('Getting prediction data...\n\n')
    viz_data = han_viz.get_visualization_data(input_docs=[sent])
    print(viz_data)
