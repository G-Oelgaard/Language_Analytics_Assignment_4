# Language_Analytics_Assignment_4
## ------ SCRIPT DESCRIPTION ------
This repository contains two script using the "VideoCommentsThreatCorpus.csv" containing toxic comments divided into threatning and non-threatning. Both scripts are train classification models on the threat corpus.

The LogReg_Class.py script will:
- Load and balance the above mentioned .csv file.
- Use either a Count or Tfidf vectorizer on the comments
- Train logistical regression classifier
- Print a classification report to show how good the classifier is.

The CNN_Class.py script will:
- Load and balance the above mentioned .csv file.
- Normalize the comments
- Tokenize, sequence and pad comments + binarize labels
- Define and train CNN classification model
- Print a classification report to show how good the classifier is.


## ------ METHODS ------


## ------ DATA ------
The data is a .csv file containing around 6335 fake and real news articles divided into the columns "title", "text" and "label".

The data was obtained through the language analytics course.

## ------ REPO STRUCTURE ------
"src" FOLDER:
- This folder contains the .py script.

"in" FOLDER:
- This is where the data used in the scripts should be placed. Ie. where the "fake_or_real_news.csv" should be placed.

"out" FOLDER:
- This is where the new .csv files will be placed.

"utils" FOLDER:
- This folder should include all utility scripts used by the main script.

## ------ SCRIPT USAGE ------
### Arguments for LogReg_Class.py script

**Required**
Argument         | What it specifies / does
---------------- | -------------------------
"-l" / "--label" | The name of the label you want to use the script on. Ie. "FAKE" or "REAL"

**Optional**
Argument         | What it specifies / does
---------------- | -------------------------
"-o" / "--output" | The filepath to the place you want to place the new .csv file in (without the output filename). If none is given the file will be outputted to the "out" folder.
"-t" / "--top" | How many of the top named entities to be printed. Ex. if given 5, the top five most common entities will be printed in the terminal

### Arguments for CNN_Class.py script

## ------ RESULTS ------
The model achieves what it sets out to do. However, the structure of the script means that it will have to be run twice to get the new results for both the fake and real news.
