# income-prediction-tensorflow
An SDSU AI Club workshop which uses Tensorflow to predict income.

The data were obtained from https://www.kaggle.com/uciml/adult-census-income


adult.csv - data from kaggle

data_parsing.py - contains a function to parse adult.csv

create_model.py - uses Tensorflow to train and save model

weights.json - the weights we determinedd through the machine learning process in create_model.py

model.py - contains a function to use the model we have trained and saved

income_prediction_app.py - the end product of our work; this is a program which utilizes the model we have trained to make predictions based on user inputted data