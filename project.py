"#!/usr/bin/env python3"
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn import tree

from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def main():
    # argv[1] /Users/monicasieklucki/Desktop/COMP379/project/Crimes_2020.csv
    # dataset_filepath = "/Users/monicasieklucki/Desktop/COMP379/project/Crimes_2020.csv"
    dataset_filepath = sys.argv[1]

    df = pd.read_csv(dataset_filepath)

    # drop all rows that contain a NaN value
    df.dropna(inplace=True)

    # remove columns that won't be needed or redundant
    df.drop(["ID"], axis = 1, inplace = True)
    df.drop(["Case Number"], axis = 1, inplace = True)
    df.drop(["Year"], axis = 1, inplace = True)
    df.drop(["Location"], axis = 1, inplace = True)
    df.drop(["Updated On"], axis = 1, inplace = True)

    # I'm having trouble converting these to float values to run my SVM/LogisticRegression classifier
    df.drop(["Block"], axis = 1, inplace = True)
    df.drop(["IUCR"], axis = 1, inplace = True)
    df.drop(["FBI Code"], axis = 1, inplace = True)
    # It might be needed for later
    df.drop(["Description"], axis = 1, inplace = True)
    df.drop(["Location Description"], axis = 1, inplace = True)

    # convert date to datetime
    df['Date']= pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    df['Hour'] = df['Date'].dt.hour
    df['Minute'] = df['Date'].dt.minute
    df['Second'] = df['Date'].dt.second

    df.drop(["Date"], axis = 1, inplace = True)

    #drop id column
    y = df["Primary Type"]
    X = df.drop("Primary Type", axis = 1)

    le = LabelEncoder()
    y = le.fit_transform(y)

    #print("-----dummy-----")
    #train a dummy classifier to make predictions based on the most_frequent class value
    #dummy_classifier_frequent = DummyClassifier(strategy="most_frequent") #baseline = most_frequent: 0.2070725616921269
    #dummy_classifier_frequent.fit(X,y)
    #dummy_classifier_frequent.predict(X)
    #print("most_frequent: " + str(dummy_classifier_frequent.score(X, y)))

    #dummy_classifier_stratified = DummyClassifier(strategy="stratified") #baseline = stratified: 0.12156041911476693
    #dummy_classifier_stratified.fit(X,y)
    #dummy_classifier_stratified.predict(X)
    #print("stratified: " + str(dummy_classifier_stratified.score(X, y)))

    #dummy_classifier_prior = DummyClassifier(strategy="prior") #prior: 0.2070725616921269
    #dummy_classifier_prior.fit(X,y)
    #dummy_classifier_prior.predict(X)
    #print("prior: " + str(dummy_classifier_prior.score(X, y)))

    #dummy_classifier_uniform = DummyClassifier(strategy="uniform") #uniform: 0.031549892283587934
    #dummy_classifier_uniform.fit(X,y)
    #dummy_classifier_uniform.predict(X)
    #print("uniform: " + str(dummy_classifier_uniform.score(X, y)))

    #dummy_classifier_constant = DummyClassifier(strategy="constant") #baseline =
    #dummy_classifier_constant.fit(X,y)
    #dummy_classifier_constant.predict(X)
    #print("constant: " + str(dummy_classifier_constant.score(X, y)))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    clf = tree.DecisionTreeClassifier()
    #'criterion':['gini','entropy'],
    #'max_features':[None, 'auto', 'sqrt', 'log2'],
    #{'max_depth': 8, 'min_samples_leaf': 3, 'min_samples_split': 4} 0.3400273882475347

    #0.3400426889444814 {'max_depth': 8, 'max_features': None, 'min_samples_leaf': 2, 'min_samples_split': 4}

    parameters = {'max_features':[None, 'auto', 'sqrt', 'log2'], 'max_depth':[2, 4, 8, 10, 11, 12, 15], 'min_samples_split':[2,4,6,8,10], 'min_samples_leaf':[1,2,3,4,5] }

    #best parameters
    #max_depth = 10

    gs_dt = GridSearchCV(clf, parameters).fit(X_train, y_train)

    print(gs_dt.best_score_)
    print(gs_dt.best_params_)
    #clf = clf.fit(X_train, y_train)

    #ypred = clf.predict(X_test)

    #accuracy = accuracy_score(y_test, ypred) #default = 0.8373983739837398
    #print("DecisionTreeClassifier Accuracy")
    #print(accuracy)





if __name__ == "__main__":
    main()
