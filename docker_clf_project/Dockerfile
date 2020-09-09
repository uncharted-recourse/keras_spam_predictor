FROM docker.uncharted.software/recourse/keras_spam_predictor_base

COPY . .

RUN pip install Flask
RUN  apt-get update \
  && apt-get install -y curl \
  && rm -rf /var/lib/apt/lists/*

RUN unzip /spam_clf_endpoint/models/jpl_augmented_hatt_rnn.h5.zip -d /spam_clf_endpoint
RUN unzip /spam_clf_endpoint/jpl_combined_keras_tokenizer.pkl.zip -d /spam_clf_endpoint

ENTRYPOINT ["/usr/local/bin/dumb-init", "--"]
CMD ["python3", "/spam_clf_endpoint/spam_clf_server.py"]
