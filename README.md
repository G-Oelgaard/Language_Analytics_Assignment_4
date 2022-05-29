# Language_Analytics_Assignment_4
## ------ SCRIPT DESCRIPTION ------
This repository contains a script that takes the dataset "fake_or_real_news.csv" and create a new dataset with either the Fake or Real news polarity and named entities. Furthermore, if the user wishes the script will print the top X named entities.

The script will:
- Create a new .csv file with either the Real or fake news.
- Append each articles polarity and named entities
- Print the top X* named entities in the terminal. 
   - '*' Specified by the user

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
### Arguments

**Required**
Argument         | What it specifies / does
---------------- | -------------------------
"-l" / "--label" | The name of the label you want to use the script on. Ie. "FAKE" or "REAL"

**Optional**
Argument         | What it specifies / does
---------------- | -------------------------
"-o" / "--output" | The filepath to the place you want to place the new .csv file in (without the output filename). If none is given the file will be outputted to the "out" folder.
"-t" / "--top" | How many of the top named entities to be printed. Ex. if given 5, the top five most common entities will be printed in the terminal

## ------ RESULTS ------
The model achieves what it sets out to do. However, the structure of the script means that it will have to be run twice to get the new results for both the fake and real news.
