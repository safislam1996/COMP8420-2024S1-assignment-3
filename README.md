# COMP8420: Advanced Natural Language Processing
## Final Project

The directory has the following files present

1.main.py: this file contains the code necessary to create the model	
2.app.py: the flask app to run the code	
3.archive: the folder containing the training and testing dataset
4.model_predict.py: model prediction to check for sentiment
5.templates: folder containing templates of the webpages


## Steps to run the code
1. Download dataset from  [here](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews) .Unzip the folder and name it `archive`. 
1. First run the `main.py` file to run the code. This produces the result folder and logs. This file prints out the necessary metrics of the model
2. Run the streamlit app. The app prompts you to give a excel sheet of reviews.
3. If the streamlit app doesnot work. Please run `model_predict.py` file for checking for sentiment





