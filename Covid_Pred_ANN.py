# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 12:59:17 2020

@author: msmit
"""
# Importing the libraries
import numpy as np
import pandas as pd
from GUI import Dialogue_Positive
from GUI import Dialogue_Negative
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler


def predict(Age,Gender,Body_Temp,Dry_Cough,Sour_Throat,Weakness,Breathing_Prob,Drowsiness,Pain_Chest,Travel_History,Diabetes,Heart_Disease,Lung_Disease,Stroke,Symptom_Progress,Blood_Press,Kidney_Disease,Appetite,Smell):
    # load json and create model
    json_file = open('model_COVID.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model_COVID.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    # Importing the dataset
    dataset = pd.read_csv('COVID.csv')
    X = dataset.iloc[:127, 0:19].values
    y = dataset.iloc[:127, 19].values

    y_list = []
    for i in range(127):
        if (y[i] == 0):
            y_list.append([1, 0, 0])
        elif (y[i] == 1):
            y_list.append([0, 1, 0])
        else:
            y_list.append([0, 0, 1])

    y = pd.DataFrame(y_list)

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # Feature Scaling

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)

    prediction = loaded_model.predict(
        sc.transform(np.array([[Age,Gender,Body_Temp,Dry_Cough,Sour_Throat,Weakness,Breathing_Prob,Drowsiness,Pain_Chest,Travel_History,Diabetes,Heart_Disease,Lung_Disease,Stroke,Symptom_Progress,Blood_Press,Kidney_Disease,Appetite,Smell]])))
    res = 0
    for i in range(len(prediction[0])):
        if (prediction[0][i] == max(prediction[0])):
            res = i
            break
    if(res==0):
        Dialogue_Negative.run()
    if(res==1):
        Dialogue_Positive.run(1)
    if(res==2):
        Dialogue_Positive.run(2)
