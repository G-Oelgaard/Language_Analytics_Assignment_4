## Importing packages ##
import os, sys, re
sys.path.append(os.path.join(".."))

import tqdm
import pandas as pd
import utils.classifier_utils as clf
import unicodedata
import contractions

# sk_learn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import classification_report
from sklearn import metrics

# tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, 
                                    Flatten,
                                    Conv1D, 
                                    MaxPooling1D, 
                                    Embedding)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.regularizers import L2

# visualisations 
import matplotlib.pyplot as plt

#args_parse
import argparse

## Functions ##
# Loading and splitting data
def load_data():
    filename = os.path.join("..","in","VideoCommentsThreatCorpus.csv") #filepath

    data = pd.read_csv(filename) #load file
    
    data_balance = clf.balance(data, 1350) #balance dataset
    
    X = data_balance["text"] # def X
    y = data_balance["label"] # def y
    
    X_train, X_test, y_train, y_test = train_test_split(X, # split
                                                        y,
                                                        test_size= 0.2,
                                                        random_state=42)
    
    label_names = ["0 / non-threat","1 / threat"] #set labels
    
    return label_names, y_test, y_train, X_test, X_train

# preprocess data
def preprocess(X_test, X_train):
    def remove_accented_chars(text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    def pre_process_corpus(docs):
        norm_docs = []
        for doc in tqdm.tqdm(docs):
            doc = doc.translate(doc.maketrans("\n\t\r", "   "))
            doc = doc.lower()
            doc = remove_accented_chars(doc)
            doc = contractions.fix(doc)
            # lower case and remove special characters\whitespaces
            doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
            doc = re.sub(' +', ' ', doc)
            doc = doc.strip()  
            norm_docs.append(doc)

        return norm_docs
    
    X_train_norm = pre_process_corpus(X_train)
    X_test_norm = pre_process_corpus(X_test)
    
    return X_test_norm, X_train_norm
    
# tokenizing, sequence, pad and binarize labels
def token(seq_length, X_train_norm, X_test_norm, y_train, y_test):

    if seq_length in locals():
        MAX_SEQUENCE_LENGTH = int(seq_length)
    else:
        MAX_SEQUENCE_LENGTH = 750
    
    t = Tokenizer(oov_token = '<UNK>')
    t.fit_on_texts(X_train_norm)
    t.word_index["<PAD>"] = 0 
    
    VOCAB_SIZE = len(t.word_index)
    
    X_train_seqs = t.texts_to_sequences(X_train_norm)
    X_test_seqs = t.texts_to_sequences(X_test_norm)
    
    X_train_pad = sequence.pad_sequences(X_train_seqs, maxlen=MAX_SEQUENCE_LENGTH, padding="post")
    X_test_pad = sequence.pad_sequences(X_test_seqs, maxlen=MAX_SEQUENCE_LENGTH, padding="post")
    
    lb = LabelBinarizer()
    y_train_lb = lb.fit_transform(y_train)
    y_test_lb = lb.fit_transform(y_test)
    
    return y_test_lb, y_train_lb, X_test_pad, X_train_pad, VOCAB_SIZE, MAX_SEQUENCE_LENGTH

# Define and run model
def CNN_model(embed_size, epoch, batch, VOCAB_SIZE, MAX_SEQUENCE_LENGTH, X_train_pad, y_train_lb):
    if embed_size in locals():
        EMBED_SIZE = int(embed_size)
    else:
        EMBED_SIZE = 300
    
    if epoch in locals():
        EPOCHS = int(epoch)
    else:
        EPOCHS = 5
    
    if batch in locals():
        BATCH_SIZE = int(batch)
    else:
        BATCH_SIZE = 128

    tf.keras.backend.clear_session() # just to be sure nothing funky happens
    
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, 
                        EMBED_SIZE, 
                        input_length=MAX_SEQUENCE_LENGTH))

    model.add(Conv1D(filters=128, 
                            kernel_size=4, 
                            padding='same',
                            activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=64, 
                            kernel_size=4, 
                            padding='same', 
                            activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=32, 
                            kernel_size=4, 
                            padding='same', 
                            activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', 
                            optimizer='adam', 
                            metrics=['accuracy'])
    model.summary()

    
    model.fit(X_train_pad, y_train_lb,
            epochs = EPOCHS,
            batch_size = BATCH_SIZE,
            validation_split = 0.1,
            verbose = True)
    
    return model

# print and save class report
def class_report(model, X_test_pad, y_test_lb, label_names, class_name):
    predictions = (model.predict(X_test_pad) > 0.5).astype("int32")
    
    class_report = classification_report(y_test_lb, predictions, target_names = label_names)
    
    print(class_report)
    
    outpath = os.path.join("..", "out", class_name)
    
    with open(outpath,"w") as file:
        file.write(str(class_report))

# args_parse
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--seq_length", required = False, help="How big do you want the sequence length to be?")
    ap.add_argument("-em", "--embed_size", required = False, help="How big do you want the embed size to be?")
    ap.add_argument("-e", "--epoch", required = False, help="How many epochs do you want the model to run?")
    ap.add_argument("-b", "--batch", required = False, help="What batchsize do you want the model to use?")
    ap.add_argument("-c", "--class_name", required = True, help="What do you want to call your classification report? Remember to include '.txt'")
    args = vars(ap.parse_args())
    return args

## Main ##
# Defining main
def main():
    args = parse_args()
    label_names, y_test, y_train, X_test, X_train = load_data()
    X_test_norm, X_train_norm = preprocess(X_test, X_train)
    y_test_lb, y_train_lb, X_test_pad, X_train_pad, VOCAB_SIZE, MAX_SEQUENCE_LENGTH = token(args["seq_length"], X_train_norm, X_test_norm, y_train, y_test)
    model = CNN_model(args["embed_size"], args["epoch"], args["batch"], VOCAB_SIZE, MAX_SEQUENCE_LENGTH, X_train_pad, y_train_lb)
    class_report(model, X_test_pad, y_test_lb, label_names, args["class_name"])

# Running main
if __name__ == "__main__":
    main()
