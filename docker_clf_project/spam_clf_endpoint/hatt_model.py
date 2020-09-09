import argparse
import json
import pickle
import numpy as np
import keras.backend as K

from keras.models import Model
from keras import initializers as initializers, regularizers, constraints
from keras.engine.topology import Layer
from keras.layers import Dense, Dropout, Embedding, GRU, Input, LSTM, Lambda
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.optimizers import Adagrad, Adam, RMSprop, SGD
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, CSVLogger

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score

from attention import AttLayer
from data import HATTData

class KerasAttModel(object):
    def __init__(self, args):
        if args.log_file is not None:
            self.log_file = open(args.log_file, encoding='utf8', mode='a')
        self.vocab_size = args.vocab_size
        self.input_length = 150
        self.recurrent_cell_map = {'gru': GRU, 'lstm': LSTM}
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.hidden_layers = args.hidden_layers
        self.bidirectional = args.bidirectional
        self.embedding_dropout_rate = args.embedding_dropout_rate
        self.recurrent_cell = self.recurrent_cell_map[args.recurrent_cell]
        self.opt = args.opt
        self.return_attention_weights = args.return_attention_weights
        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.tokenizer = Tokenizer(num_words=self.vocab_size+1, filters='!"#$%&()*+,-.:;=?@[\\]^_`{|}~\t\n',
                                   lower=True, split=' ', oov_token="<UNK>")
        self._choose_optimizer()
        self.build_model()
        self._log_params()

    def _choose_optimizer(self):
        if self.opt == 'adam':
            self.optimizer = Adam()
        elif self.opt == 'rmsprop':
            self.optimizer = RMSprop()
        elif self.opt == 'adagrad':
            self.optimizer = Adagrad()
        elif self.opt == 'sgd':
            self.optimizer = SGD(lr=0.005, momentum=0.99, decay=0.9)

    def _init_layers(self):
        self.embedding_layer = Embedding(input_dim=self.vocab_size + 1, output_dim=self.embedding_dim, mask_zero=False)
        self.recurrent_cells = [Bidirectional(self.recurrent_cell(units=self.hidden_dim, return_sequences=True)) if self.bidirectional
                                else self.recurrent_cell(units=self.hidden_dim, return_sequences=True) for _ in range(self.hidden_layers)]
        self.attention_layer = AttLayer(attention_dim=self.hidden_dim)
        self.out_layer = Dense(units=1, activation='sigmoid')

    def _log_params(self):
        print()
        print('TRAINING PARAMETERS')
        print('=' * 50)
        print('Recurrent cell:', self.recurrent_cell)
        print('Embedding dropout:', self.embedding_dropout_rate)
        print('Optimizer:', self.optimizer)
        print('Embedding dim:', self.embedding_dim)
        print('Hidden dim:', self.hidden_dim)
        print('Num hidden layers:', self.hidden_layers)
        print('Training epochs:', self.n_epochs)
        print('Batch size:', self.batch_size)
        print()

    def build_model(self):
        self._init_layers()
        in_layer = Input(shape=(None,))
        embedded = self.embedding_layer(in_layer)
        if self.embedding_dropout_rate > 0.:
            embedded = Dropout(rate=self.embedding_dropout_rate)(embedded)
        for layer_idx, layer in enumerate(self.recurrent_cells):
            if layer_idx == 0:
                h_out = layer(embedded)
            else:
                h_out = layer(h_out)
        
        # x, attn = self.attention_layer(h_out)
        # x = Dropout(rate=0.5)(x)
        # x = Lambda(lambda x: K.max(x, axis=1))(h_out)
        x = AttLayer(attention_dim=self.hidden_dim)(h_out)

        y_out = self.out_layer(x)
        model = Model(inputs=in_layer, outputs=y_out)
        model.summary()
        self.model = model
    
    def _get_sequences(self, texts):
        seqs = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(seqs, maxlen=self.input_length,
                             value=0, padding='post', truncating='post')

    def _get_attention_map(self, texts):
        att_model_output = self.model.layers[0:-2]
        att_model = Model(att_model_output[0].input, att_model_output[-1].output)
        att_model.compile(optimizer=self.optimizer,
                          loss="binary_crossentropy",
                          metrics=["accuracy"])
        return att_model.predict(self._get_sequences(texts))[1]

    def _report_metrics(self, epoch, y_test, y_hat, y_score=None):
        report_str = """
        EPOCH {} METRICS
        =====================
        Accuracy: {}
        ROC-AUC: {}
        Avg Precision: {}
        Precision: {}
        Recall: {}
        F1: {}
        """

        report_str_populated = report_str.format(
            epoch,
            accuracy_score(y_test, y_hat),
            roc_auc_score(y_test, y_score),
            average_precision_score(y_test, y_score),
            precision_score(y_test, y_hat),
            recall_score(y_test, y_hat),
            f1_score(y_test, y_hat)
        )

        print(report_str_populated)
        self.log_file.write(report_str_populated)

    def train(self, X_train, y_train, X_val=None, y_val=None, class_weight=None):
        np.random.seed(7)

        # Compile model for training
        self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        self.tokenizer.fit_on_texts(X_train)

        # I think the lines below may be buggy/unnecessary - note-to-self: look at this again when you get a second
        # self.tokenizer.word_index = {e: i for e,i in self.tokenizer.word_index.items() if i <= self.vocab_size}
        # self.tokenizer.word_index[self.tokenizer.oov_token] = self.vocab_size + 1

        # Saving tokenizer object
        with open('keras_tokenizer.pkl', mode='wb') as outfile:
            pickle.dump(self.tokenizer, outfile, pickle.HIGHEST_PROTOCOL)

        print('Tokenizer saved...')

        X_train = self._get_sequences(X_train)
        print('X_train shape:')
        print(X_train.shape)

        # Setup checkpoint files for dynamic model-saving
        ckpt_filename = "models/att_rnn_model-epoch{:02d}-{:.2f}.h5"
        # ckpt = ModelCheckpoint(ckpt_filename, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        if X_val is not None:
            X_val = self._get_sequences(X_val)
            # Manually write the epoch loop so we can get P/R reports
            for e in range(self.n_epochs):
                hist = self.model.fit(X_train, y_train, validation_data=[X_val, y_val],
                                      batch_size=self.batch_size, epochs=1, class_weight=class_weight)
                y_score = self.model.predict(X_val, batch_size=self.batch_size).flatten()
                y_hat = np.asarray([1 if score > 0.5 else 0 for score in y_score])
                self._report_metrics(epoch=e + 1, y_test=y_val, y_hat=y_hat, y_score=y_score)
                self.model.save(filepath=ckpt_filename.format(e + 1, np.mean(hist.history['val_acc'])))
        else:
            self.model.fit(X_train, y_train, batch_size=self.batch_size,
                           epochs=self.n_epochs, validation_split=0.1)
        
        self.log_file.close()


class HattKerasModel(KerasAttModel):
    def __init__(self, args, mode='train'):
        self.train_file = args.train_file
        self.valid_file = args.valid_file
        self.top_hidden_dim = args.top_hidden_dim
        super().__init__(args)
        self.data_obj = HATTData(data_file=self.train_file, max_sents=50, 
                                 max_len=200, batch_size=self.batch_size, mode=mode)
        self.num_train_examples = 441708 # Hard-coded for now...
        self.num_valid_examples = 63102 # Hard-coded for now...
        self.num_train_iters = self.num_train_examples // self.batch_size
        self.num_valid_iters = self.num_valid_examples // self.batch_size

    def _init_layers(self):
        # Sentence-level model
        self.sent_input = Input(shape=(None,))
        self.sent_embedding = Embedding(input_dim=self.vocab_size + 1, output_dim=self.embedding_dim)
        self.sent_recurrent_cells = [Bidirectional(self.recurrent_cell(units=self.hidden_dim, return_sequences=True)) if self.bidirectional
                                else self.recurrent_cell(units=self.hidden_dim, return_sequences=True) for _ in range(self.hidden_layers)]
        self.sent_att_layer = AttLayer(attention_dim=self.hidden_dim)

        # Document-level model
        self.doc_input = Input(shape=(None, None))
        self.doc_recurrent_cells = [Bidirectional(self.recurrent_cell(units=self.top_hidden_dim, return_sequences=True)) if self.bidirectional
                                else self.recurrent_cell(units=self.top_hidden_dim, return_sequences=True) for _ in range(self.hidden_layers)]
        self.doc_att_layer = AttLayer(attention_dim=self.top_hidden_dim)
        self.out_layer = Dense(units=1, activation='sigmoid')

    def build_model(self):
        self._init_layers()
        # Build sentence-level encoder
        sent_embedded = self.sent_embedding(self.sent_input)
        for idx, layer in enumerate(self.sent_recurrent_cells):
            if idx == 0:
                h_sent_out = layer(sent_embedded)
            else:
                h_sent_out = layer(h_sent_out)
        
        h_att_sent = self.sent_att_layer(h_sent_out)
        sent_encoder = Model(inputs=self.sent_input, outputs=h_att_sent)

        # Build hierarchical encoder
        doc_encoder = TimeDistributed(sent_encoder)(self.doc_input)
        for idx, layer in enumerate(self.doc_recurrent_cells):
            if idx == 0:
                h_out = layer(doc_encoder)
            else:
                h_out = layer(h_out)
        
        h_att_doc = self.doc_att_layer(h_out)
        y_out = self.out_layer(h_att_doc)

        model = Model(inputs=self.doc_input, outputs=y_out)
        model.summary()
        self.model = model

    def train(self):
        np.random.seed(7)
        self.model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['acc'])

        # Setup checkpoint files for dynamic model-saving
        ckpt_filename = "models/bidi_hatt_rnn_model-epoch{:02d}-loss{:.2f}.h5"

        for epoch in range(self.n_epochs):
            train_datagen = self.data_obj.data_generator(data_file=self.train_file)
            valid_datagen = self.data_obj.data_generator(data_file=self.valid_file)
            
            self.model.fit_generator(train_datagen, steps_per_epoch=self.num_train_iters,
                                     class_weight={0: 0.03, 1: 0.97})
            valid_loss, valid_acc = self.validate(valid_generator=valid_datagen, epoch=epoch)
            print('Valid acc:', valid_acc)
            self.model.save(filepath=ckpt_filename.format(epoch + 1, valid_loss))

    def validate(self, valid_generator, epoch):
        y_test, preds, scores, losses = [], [], [], []
        for _ in range(self.num_valid_iters):
            x_valid_batch, y_valid_batch = next(valid_generator)
            losses.append(self.model.test_on_batch(x_valid_batch, y_valid_batch))
            y_score = list(self.model.predict_on_batch(x_valid_batch).flatten())
            y_hat = [1 if score > 0.5 else 0 for score in y_score]
            scores.extend(y_score)
            preds.extend(y_hat)
            y_test.extend(list(y_valid_batch))

        self._report_metrics(epoch=epoch, y_test=y_test, y_hat=preds, y_score=scores)
        mean_loss = sum([i[0] for i in losses]) / len(losses)
        mean_acc = sum([i[1] for i in losses]) / len(losses)
        return mean_loss, mean_acc

# Test compiling model
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str, required=False, default=None)
    parser.add_argument('--embedding_dim', type=int, required=False, default=128)
    parser.add_argument('--hidden_dim', type=int, required=False, default=64)
    parser.add_argument('--top_hidden_dim', type=int, required=False, default=0)
    parser.add_argument('--hidden_layers', type=int, required=False, default=1)
    parser.add_argument('--embedding_dropout_rate', type=float, required=False, default=0.2)
    parser.add_argument('--recurrent_cell', type=str, required=False, default='gru')
    parser.add_argument('--bidirectional', type=bool, required=False, default=False)
    parser.add_argument('--return_attention_weights', type=bool, required=False, default=True)
    parser.add_argument('--vocab_size', type=int, required=False, default=100000)
    parser.add_argument('--opt', type=str, required=False, default='adam')
    parser.add_argument('--n_epochs', type=int, required=False, default=10)
    parser.add_argument('--batch_size', type=int, required=False, default=128)
    args = parser.parse_args()

    if args.top_hidden_dim == 0:
        att_model = KerasAttModel(args)
        print('Model successfully built!')
    else:
        print('Building hierarchical model...')
        hatt_model = HattKerasModel(args)
        print('Hierarchical model successfully built!')
