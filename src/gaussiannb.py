#!/usr/bin/python

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import confusion_matrix
import json

from sklearn.naive_bayes import GaussianNB

infile = "../json/hr-sz13-rk08-results.json"

def unwrap_line(line):
    result = [] # leave off identifying string
    for i in range(1, 13):
        if line[i] == "True":
            result.append(True)
        else:
            result.append(False)
    for i in range(13, 15):
        result.append(int(line[i]))
    return result

def graphic_check(line):
    """
    A binary matroid is graphic iff it has no F_7, K_5, or K_3,3 minor
    Check if columns 1, 3, 5 are all False
    """
    return not (line[0] or line[1] or line[3] or line[5])


def preprocess(datafile = infile):
    with open(datafile) as f:
        data_strings = json.load(f)
    data = []

    for line in data_strings:
        data.append(unwrap_line(line))
    print("number of matroids:", len(data))

    is_graphic = [graphic_check(line) for line in data]

    # remove certain attributes from data
    # without this, the model will work with 100% accuracy
    # allowed_attribs = [2, 4] + list(range(6, 14))
    allowed_attribs = [0, 3, 5]
    data = [[line[i] for i in allowed_attribs] for line in data]

    ### test_size is the percentage of events assigned to the test set
    ### (remainder go into training)
    matroids_train, matroids_test, labels_train, labels_test = train_test_split(
        data, is_graphic, test_size=0.2, random_state=42)

    print(matroids_test[0])
    print("how many ambiguous?")
    print(sum(1 for line in matroids_test if not (line[0] or line[1] or line[2])))
    print(len([i for i in range(len(matroids_test)) if sum(matroids_test[i]) == 0]))


    # # let it discard a redundant feature or two
    # selector = SelectPercentile(f_classif, percentile=90)
    # selector.fit(matroids_train, labels_train)
    # matroids_train_transformed = selector.transform(matroids_train)
    # matroids_test_transformed  = selector.transform(matroids_test)

    ### info on the data
    print("Number of graphic matroids in training data:", sum(labels_train))
    print("Number of nongraphic matroids in training data:", len(labels_train)-sum(labels_train))
    print("Number of graphic matroids in test data:", sum(labels_test))
    print("Number of nongraphic matroids in test data:", len(labels_test)-sum(labels_test))

    return matroids_train, matroids_test, labels_train, labels_test
    # return matroids_train_transformed, matroids_test_transformed, labels_train, labels_test

def predict_if_graphic(matroids_train, matroids_test, labels_train, labels_test):
    clf = GaussianNB()
    clf.fit(matroids_train, labels_train)
    accuracy = clf.score(matroids_test, labels_test)
    print(len(matroids_test))
    print(len(matroids_test[0]))
    print(confusion_matrix(labels_test, clf.predict(matroids_test)))
    print()
    print(accuracy)

    print("number of dumb false positives")
    print(matroids_test[0])
    print(len([i for i in range(len(matroids_test)) if sum(matroids_test[i]) == 0 and clf.predict(matroids_test)[i]]))
    return accuracy

def simple_statistics(datafile = "../json/hr-sz13-rk08-results.json"):
    with open(datafile) as f:
        data_strings = json.load(f)
    data = [unwrap_line(line) for line in data_strings]

    print("ratio connected:", (sum(data[9]) / len(data)))
    print(sum(1 for line in data if (line[1] and not (line[0] or line[3] or line[5]))))


if __name__ == "__main__":
    matroids_train, matroids_test, labels_train, labels_test = preprocess()
    predict_if_graphic(matroids_train, matroids_test, labels_train, labels_test)
    simple_statistics()
