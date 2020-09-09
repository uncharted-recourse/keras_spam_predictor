import argparse
import numpy as np
import pandas as pd
import pickle

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize, sent_tokenize


class HATTData(object):
    def __init__(self, data_file:str, max_sents:int, max_len:int, batch_size:int, mode:str='train'):
        self.max_sents = max_sents
        self.max_len = max_len
        self.data_file = data_file
        self.vocab_size = 70000
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<UNK>')
        self.batch_size = batch_size
        if mode == 'train':
            self._get_vocab()

    def _get_vocab(self):
        df = pd.read_csv(self.data_file, sep='\t', encoding='utf8')
        print('DataFrame loaded...')
        print(df.head())
        all_text = [str(d) for d in df.text.tolist()]
        self.tokenizer.fit_on_texts(all_text)
        print('Finished fitting tokenizer')

        tok_path = 'bidi_hatt_tokenizer.pkl'
        with open(tok_path, 'wb') as outfile:
            pickle.dump(self.tokenizer, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        
        print('Tokenizer saved to {}!'.format(tok_path))

    def _tokenize(self, input_sent:str):
        w_ids = self.tokenizer.texts_to_sequences([input_sent])
        w_ids_padded = pad_sequences(w_ids, maxlen=self.max_len,
                                     padding='post', value=0)
        return w_ids_padded[0]

    def parse_hierarchical_emails(self, input_df:pd.DataFrame):
        text = input_df.text.tolist()
        labels = input_df.label.tolist()
        emails = []

        for idx, email_body in enumerate(text):
            sys.stdout.write('\rProcessing email {}...'.format(idx + 1))
            emails.append(sent_tokenize(email_body.strip()))

        return text, labels, emails

    def emails_to_data(self, emails:list, text:list):
        self.tokenizer.fit_on_texts(text)
        X = np.zeros((len(text), self.max_sents, self.max_len), dtype='int32')

        for i, email_sent_blob in enumerate(emails):
            for j, sent in enumerate(email_sent_blob):
                sent_word_ids = self._tokenize(sent)
                for k, w_id in enumerate(sent_word_ids):
                    X[i, j, k] = w_id

        return X

    def _pad_sentence(self, sentence:list, max_num_words:int):
        word_diff = max_num_words - len(sentence)
        padding_words = [0] * word_diff
        sentence.extend(padding_words)
        return sentence

    def pad_data(self, x_batch:list, seq_lengths:list, num_sents:list):
        # Each batch should have shape (batch_size, max_num_sents, max_num_words)
        # batch_size is still number of docs
        max_num_words = max(seq_lengths)
        if max(num_sents) < self.max_sents:
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

    def data_generator(self, data_file:str):
        while True:
            x_batch, y_batch, seq_lengths, num_sents = [], [], [], []
            with open(data_file, encoding='utf8', mode='r') as infile:
                for idx, line in enumerate(infile):
                    # Skip headers in the TSV
                    if idx == 0:
                        continue
                    x, y = line.strip().split('\t')
                    # Get sentences within doc - we assume data has been preprocessed
                    # with inline sentence markers
                    x_sents = [s.strip() for s in x.split('|')] # sent_tokenize(x.strip())
                    num_sents.append(len(x_sents))
                    x_sent_toks = self.tokenizer.texts_to_sequences(x_sents)
                    # Trim extra-long sentences
                    x_sent_toks = [s[:self.max_len] for s in x_sent_toks]
                    seq_lengths.append(max([len(s) for s in x_sent_toks]))
                    # Push to batch containers
                    # Try trimming number of sents to push to x_batch here
                    x_batch.append(x_sent_toks[:self.max_sents])
                    if '.' in y:
                        y_batch.append(int(float(y)))
                    else:
                        y_batch.append(int(y))
                    # Check batch size for padding and yielding
                    if len(x_batch) == self.batch_size:
                        x_batch_padded = self.pad_data(x_batch, seq_lengths, num_sents)
                        yield x_batch_padded, np.asarray(y_batch)
                        x_batch, y_batch, seq_lengths, num_sents = [], [], [], []
                        # yield x_batch, y_batch, seq_lengths, num_sents
                        # x_batch, y_batch, seq_lengths, num_sents = [], [], [], []
                
                if len(x_batch) > 0:
                    x_batch_padded = self.pad_data(x_batch, seq_lengths, num_sents)
                    yield x_batch_padded, np.asarray(y_batch)


if __name__ == '__main__':
    import os
    import sys
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, required=False, default='/data/users/kyle.shaffer/ased_data/ham_spam_processed.tsv')
    parser.add_argument('--max_sents', type=int, required=False, default=50)
    parser.add_argument('--max_len', type=int, required=False, default=200)
    parser.add_argument('--batch_size', type=int, required=False, default=64)
    args = parser.parse_args()

    hatt_data = HATTData(data_file=args.data_file, max_sents=args.max_sents, max_len=args.max_len, batch_size=args.batch_size)

    datagen = hatt_data.data_generator(args.data_file)
    for _ in range(10):
        x, y = next(datagen)
        print('x shape:', x.shape)
        print('y shape:', y.shape)
