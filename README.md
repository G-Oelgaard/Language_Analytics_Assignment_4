# Language_Analytics_Assignment_4
## ------ SCRIPT DESCRIPTION ------
This repository contains a script that takes a user-defined image and findes the three closest images based on difference in histograms.

The model will:
- print the names of the three closest images in the console.
- create a .csv with the image names of the three closest images
- A .jpg plot of the target image and the three closest images + their hist scores.

## ------ METHODS ------
To compare the histograms of images the functions imread, cvtColor, calcHist, normalize and compareHist from the cv2 package are used. Especially "compareHist" is essential when we want to compare two histograms.

To calculate the distances from our target image while still retaining our image_id and image_data, the code effectively creates a list within a list. Then to find the top 3 closest images, we sort the lists by their image_score.

## ------ DATA ------
The data is a large collection of flower images provided through the course.

## ------ REPO STRUCTURE ------
"src" FOLDER:
- This folder contains the .py script to find the closest image based on histogram and create plot + .csv

"in" FOLDER:
- This is where the data used in the scripts should be placed. Ie. where the flowers should be placed in a folder called "flowers".

"out" FOLDER:
- This is where the .csv and image plot will be saved

"utils" FOLDER:
- This folder should include all utility scripts used by the main script.

## ------ SCRIPT USAGE ------
- The model_creation.py script requires you to give the arguments "-i" / "--image" (the name of the user-defined image. Don't include .jpg)

- The poster_prediction.py script requires you to give the argument "-c" / "--csv" (the name of the csv created by the script. Don't include .csv)

- The poster_prediction.py script requires you to give the argument "-p" / "--plot" (the name of the image plot file created by the script)

## ------ RESULTS ------
The model achieves what it sets out to do. However, we might quickly run into problems if the script is used to find similar images in the eyes of a human. That is because while the histograms (in other words the color concentrations and values) of the images might be similar, what the images actually depict might not be. That is why finding similar images is probably better achieved using something like feature extraction and the nearest_neighbor function from sklearn.
