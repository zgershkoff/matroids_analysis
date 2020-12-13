#!/usr/bin/python

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import json
import sys

from sklearn.naive_bayes import GaussianNB

# for testing
infile = "../json/hr-sz13-rk08-results.json"

def unwrap_line(line):
    # line is a tuple (representation_string, attributes)
    attributes = line[1]
    return attributes

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

    num_attribs = len(data[0])
    is_graphic = [graphic_check(line) for line in data]

    # remove certain attributes from data
    # without this, the model will work with 100% accuracy
    allowed_attribs = [0, 1, 2, 4] + list(range(6, num_attribs))
    # here 3 and 5 are removed, meaning K_5 and K_3,3 duals
    data = [[line[i] for i in allowed_attribs] for line in data]

    ### test_size is the percentage of events assigned to the test set
    ### (remainder go into training)
    matroids_train, matroids_test, labels_train, labels_test = train_test_split(
        data, is_graphic, test_size=0.2, random_state=42)

    print("Example event:")
    print(matroids_test[0])

    ### info on the data
    print("Number of graphic matroids in training data:", sum(labels_train))
    print("Number of nongraphic matroids in training data:", len(labels_train)-sum(labels_train))
    print("Number of graphic matroids in test data:", sum(labels_test))
    print("Number of nongraphic matroids in test data:", len(labels_test)-sum(labels_test))

    return matroids_train, matroids_test, labels_train, labels_test

def predict_if_graphic(matroids_train, matroids_test, labels_train, labels_test):
    clf = GaussianNB()
    clf.fit(matroids_train, labels_train)
    accuracy = clf.score(matroids_test, labels_test)

    # First row is nongraphic matroids
    # First column is matroids predicted to be nongraphic
    print(confusion_matrix(labels_test, clf.predict(matroids_test)))

    # With 0 and 1 removed, the important parameters 3 and 5 have shifted
    # to indices 1 and 3
    print("number of dumb positives")
    print(len([i for i in range(len(matroids_test)) if \
     not matroids_test[i][1] and not matroids_test[i][3] \
     and clf.predict(matroids_test)[i]]))
    print(accuracy)

def simple_statistics(datafile = infile):
    """
    returns simple statistics about the matroid

    INPUT: A file path
    OUTPUT: None. Prints statistics.
    """
    with open(datafile) as f:
        data_strings = json.load(f)
    data = [unwrap_line(line) for line in data_strings]

    print("ratio connected:", (sum(data[9]) / len(data)))
    print("Number of graphic matroids:",
        sum(1 for line in data if (line[1] \
            and not (line[0] or line[3] or line[5]))))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        infile = sys.argv[1]
    matroids_train, matroids_test, labels_train, labels_test = preprocess(infile)
    predict_if_graphic(matroids_train, matroids_test, labels_train, labels_test)
    simple_statistics(infile)
