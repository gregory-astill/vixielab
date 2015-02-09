from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np

def dataprep(filespot, deplabel, testsize):
    try:
        data = pd.read_csv(filespot)
    except:
        try:
            data = pd.read_csv(filespot)
        except:
            print "Data not in CSV or Excel format."
            return None, None, None, None

    try:
        y = data[deplabel]
        x = data.drop(deplabel, 1)
    except:
        print "Invalid Dependent Variable"
        return None, None, None, None

    if ((testsize <=1) & (testsize >=0)):
        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = testsize)
        return x_train, x_test, y_train, y_test
    else:
        print "Invalid Proportion for Test Set"

def runGaussian(x_train, x_test, y_train, y_test):
    clf = GaussianNB()
    try:
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)

        acc = accuracy_score(y_test, pred)

        print "Accuracy Score:" + str(acc)

    except:
        print "Invalid Data Provided for Naive Bayes"
        return 0
    return clf
