#
# GRPC Server for Keras Spam Classifier
# 
# Uses GRPC service config in protos/grapevine.proto
# 

import os
import sys
import numpy as np
import pickle
import tensorflow as tf
from concurrent import futures
import time
import logging
import grpc
import json
from flask import Flask


# Keras utils
from attention_visualizer import *

import grapevine_pb2
import grapevine_pb2_grpc


restapp = Flask(__name__)


# GLOBALS
GRPC_PORT = '50051'

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

BASE_DIR = os.getcwd()
print('Current working directory:', BASE_DIR)

MODEL_PATH = os.path.join(BASE_DIR, 'spam_clf_endpoint/models/jpl_augmented_hatt_rnn.h5')
print(MODEL_PATH)

TOKENIZER_PATH = os.path.join(BASE_DIR, 'spam_clf_endpoint/jpl_combined_keras_tokenizer.pkl')
print(TOKENIZER_PATH)
CLASS_MAP = {0: 'ham', 1: 'spam'}

with open(TOKENIZER_PATH, 'rb') as infile:
    TOKENIZER = pickle.load(infile)

#-----
class KerasSpamClassifier(grapevine_pb2_grpc.ClassifierServicer):

    # Main classify function
    def Classify(self, request, context):

        # init classifier result object
        result = grapevine_pb2.Classification(
            domain='attack',
            prediction='false',
            confidence=0.0,
            model="qntfy_attention_rnn",
            version="0.0.1",
            meta=grapevine_pb2.Meta(),
        )

        # global graph
        with graph.as_default():
            # get text from input message
            input_doc = request.text
            
            # Exception cases
            if (len(input_doc.strip()) == 0) or (input_doc is None):
                return result

            prediction_data = MODEL_OBJECT.get_visualization_data(input_docs=[input_doc])

            if prediction_data['doc_prediction'] == 'SPAM':
                result.prediction = 'true'

            result.confidence = prediction_data['doc_score']

            if 'sentences' in prediction_data:
                clf_metadata = self.format_metadata(prediction_data['sentences'])
                result.meta.CopyFrom(clf_metadata)

            print(prediction_data)

            return result


    # Convert classifier 'sentences' metadata into required protobuf format
    def format_metadata(self, sentences):

        thisMeta = grapevine_pb2.Meta()
        sentList = []
        for s in sentences:
            thisSent = grapevine_pb2.Sentence()
            thisSent.sentence_score = s['sent_score']
            thisSent.words.extend(s['words'])
            thisSent.word_scores.extend(s['word_scores'])

            sentList.append(thisSent)

        thisMeta.sentences.extend(sentList)

        return thisMeta


#-----
def load_model_obj():
    global graph
    global MODEL_OBJECT

    MODEL_OBJECT = HierarchicalAttentionViz(model_path=MODEL_PATH, tokenizer=TOKENIZER)

    graph = tf.get_default_graph()  


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    grapevine_pb2_grpc.add_ClassifierServicer_to_server(KerasSpamClassifier(), server)
    server.add_insecure_port('[::]:' + GRPC_PORT)
    server.start()
    restapp.run()

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

@restapp.route("/healthcheck")
def health():
    return "HEALTHY"


if __name__ == '__main__':
    logging.basicConfig()
    load_model_obj()        # Load model
    serve()