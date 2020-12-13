# girth prediction with support vector machines

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import json
import sys

from sklearn.svm import SVC, SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict

# for testing
infile = "../json/hr-sz13-rk08-results.json"

# I want to rewrite how I load the data file, so I can refer to the
# features by name instead of by index. Ultimately it would probably
# be better to implement this during preprocessing.
def preprocess(datafile = infile):
    with open(infile, 'r') as f:
        data_tuples = json.load(f)

    feature_labels = ["fano", "fano_dual", "K33", "K33_dual", "K5", "K5_dual",
        "triangle", "cosimple", "conn", "3conn", "4conn", "girth", "cogirth"]
    data_w_labels = []
    for _, features in data_tuples:
        data_w_labels.append(OrderedDict({feature_labels[i]: features[i] \
            for i in range(len(features))}))
    # This allows me to refer to features by name.

    excluded = ["girth", "triangle"]
    data = [[d[key] for key in d if key not in excluded] for d in data_w_labels]
    girth = [d["girth"] for d in data_w_labels]

    ### test_size is the percentage of events assigned to the test set
    ### (remainder go into training)
    matroids_train, matroids_test, labels_train, labels_test = train_test_split(
        data, girth, test_size=0.2, random_state=42)

    print("Example event:")
    print(matroids_train[0])

    print("Possible girths", set(girth))

    return matroids_train, matroids_test, labels_train, labels_test

def svm_classifier(matroids_train, matroids_test, labels_train, labels_test, C = 1):
    clf = SVC(kernel="rbf", C=C)
    clf.fit(matroids_train, labels_train)

    pred = clf.predict(matroids_test)

    print("Accuracy and confusion matrix:")
    print(clf.score(matroids_test, labels_test))
    print(confusion_matrix(labels_test, clf.predict(matroids_test)))

def svm_regressor(matroids_train, matroids_test, labels_train, labels_test, C = 1):
    regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
    regr.fit(matroids_train, labels_train)

    pred = regr.predict(matroids_test)

    print("Accuracy:")
    print(regr.score(matroids_test, labels_test))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        infile = sys.argv[1]
    print(infile)
    print("hi")
    matroids_train, matroids_test, labels_train, labels_test = preprocess(infile)
    svm_classifier(matroids_train, matroids_test, labels_train, labels_test)
