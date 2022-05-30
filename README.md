# Language_Analytics_Assignment_4
## ------ SCRIPT DESCRIPTION ------
This repository contains two scripts using the "VideoCommentsThreatCorpus.csv" containing toxic comments divided into threatening and non-threatening. Both scripts are train classification models on the threat corpus.

The LogReg_Class.py script will:
- Load and balance the above mentioned .csv file.
- Use either a Count or Tfidf vectorizer on the comments
- Train logistic regression classifier
- Print a classification report to show how good the classifier is.

The CNN_Class.py script will:
- Load and balance the above mentioned .csv file.
- Normalize the comments
- Tokenize, sequence and pad comments + binarize labels
- Define and train CNN classification model
- Print a classification report to show how good the classifier is.

## ------ DATA ------
The data is a .csv file containing 28643 comments with the labels "0" or "1". As indicated by the research papers that created the dataset. "0" is non-threatening comments and "1" is threatening comments.

The data was obtained through the language analytics course.

The research papers behind the data:
- Hammer, H. L., Riegler, M. A., Øvrelid, L. & Veldal, E. (2019). "THREAT: A Large Annotated Corpus for Detection of Violent Threats". 7th IEEE International Workshop on Content-Based Multimedia Indexing.
- Wester, A. L., Øvrelid, L., Velldal, E., & Hammer, H. L. (2016). "Threat detection in online discussions". Proceedings of the 7th Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis.

## ------ REPO STRUCTURE ------
"src" FOLDER:
- This folder contains the .py scripts.

"in" FOLDER:
- This is where the data used in the scripts should be placed. Ie. where the "VideoCommentsThreatCorpus.csv" should be placed.

"out" FOLDER:
- This is where the classification reports will be placed.

"utils" FOLDER:
- This folder should include all utility scripts used by the main script.

## ------ SCRIPT USAGE ------
### Arguments for LogReg_Class.py script:
**Required**
Argument         | What it specifies / does
---------------- | -------------------------
"-v" / "--vector" | What type of vectorizer you want to use. Either 'count' or 'tfidf'?
"-c" / "--class_name" | What you want your classification report to be named. Remember to include '.txt'


### Arguments for CNN_Class.py script:
Argument         | What it specifies / does
---------------- | -------------------------
"-c" / "--class_name" | What you want your classification report to be named. Remember to include '.txt'

**Optional**
Argument         | What it specifies / does
---------------- | -------------------------
"-s" / "--seq_length" | How big you want the sequence length to be. 750 if not specified.
"-em" / "--embed_size" | How big you want the embed size to be. 300 if not specified. 
"-e" / "--epoch" | How many epochs you want the model to run. 5 if not specified.
"-b" / "--batch" | What batchsize you want the model to use. 128 if not specified. 

## ------ RESULTS ------
The scripts achieve what they set out to do. The classification reports also show a clear improvement when using a CNN model compared to a LogReg model. As the time it took to run the CNN model was not much longer than the LogReg model, it would in almost all cases be better to use that model for relevant predictions. 
