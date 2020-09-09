# keras_spam_predictor
Simple code and model for making binary predictions on the spam/ham classification task.

Before running code both the `*rnn.ht.zip` and `*tokenizer.zip` files need to be unzipped (into same location as respective zip files)

## Dockerized gRPC implemenation of classifier

Build docker image by docker-compose in ./docker_clf_project dir:  `docker-compose build`

Run the container using: `docker run -it -p 50051:50051 <docker image id>`
By default, gRPC connection is via port 50051

Test the gRPC server. In a separate command window run the script `./docker_clf_project/spam_clf_endpoint/spam_clf_client.py`
This sends two separate messages to the server. The first should be classified as HAM (score 0.0), and the 2nd as SPAM (score 1.0).

## Protobuf Configuration

gRPC protobuf config is in the grapevine.proto file. If this configuration is changed, follow instructions in `./docker_clf_project/spam_clf_endpoint/readme.md` to re-generate the necessary python protobuf files.

## Command-line version of classifier

You can use the `inference.py` script as a command-line tool to save out predictions for your files. By default, a tab-separated file will be written to disk (named `preds.csv`) where each row is indexed by the filename that predictions were generated for. The script also generates class predictions as well as probability or "confidence value" for predictions by default for each file. 

To run the script, clone the repo and `cd` to `keras_spam_predictor` directory. There are several command-line arguments that can be passed to the script, but the only required one is `--data_dir` which points to a directory of plain-text files. The script assumes that each plain-text file within the `data_dir` directory corresponds to one document that should be classified by the model.

To run a demo using 3 dummy files under the supplied `files` directory, simple run the following command at the command-line:

`python inference.py --data_dir=files`

This should produce a CSV `preds.csv` with 3 rows and corresponding predictions.
