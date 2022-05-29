## Importing packages ##
import os, sys
sys.path.append(os.path.join(".."))

import tqdm
import pandas as pd
import utils.classifier_utils as clf
import unicodedata
import contractions

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report
from sklearn import metrics

# matplot 
import matplotlib.pyplot as plt

#args_parse
import argparse

## Functions ##
# Loading and splitting data
def load_data():
    filename = os.path.join("..", "in", "VideoCommentsThreatCorpus.csv") #filepath

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

# Choosing vectorizer
def vectorize(vector, X_train, X_test):
    if vector in ["count", "Count"]:
        print("Classification using a CountVectorizer:")
        vectorizer = CountVectorizer(ngram_range=(1,1),
                             lowercase=True,
                             max_df=0.95,
                             min_df=0.05,
                             max_features=100)
    else:
        print("Classification using a TfidfVectorizer:")
        vectorizer = TfidfVectorizer(ngram_range=(1,1),
                             lowercase=True,
                             max_df=0.95,
                             min_df=0.05,
                             max_features=100)
    
    X_train_feats = vectorizer.fit_transform(X_train)
    X_test_feats = vectorizer.transform(X_test)
    
    return X_test_feats, X_train_feats

# running logReg classifier
def logReg_classifier(X_train_feats, y_train, X_test_feats, y_test, label_names, class_name):
    classifier = LogisticRegression(random_state = 42).fit(X_train_feats,y_train)
    
    y_pred = classifier.predict(X_test_feats)
    class_report = metrics.classification_report(y_test, y_pred, target_names = label_names)
    
    print(class_report)
    
    outpath = os.path.join("..", "out", class_name)
    
    with open(outpath,"w") as file:
        file.write(str(class_report))

# args_parse
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--vector", required = True, help="What type of vectorizer do you want to use? 'count' or 'tfidf'?")
    ap.add_argument("-c", "--class_name", required = True, help="What do you want to call your classification report? Remember to include '.txt'")
    args = vars(ap.parse_args())
    return args

## Main ##
# Defining main
def main():
    args = parse_args()
    label_names, y_test, y_train, X_test, X_train = load_data()
    X_test_feats, X_train_feats = vectorize(args["vector"], X_train, X_test)
    logReg_classifier(X_train_feats, y_train, X_test_feats, y_test, label_names, args["class_name"])

# Running main
if __name__ == "__main__":
    main()