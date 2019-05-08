from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def SVM(X_train, y_train, X_test, y_test):
    """ Classify test set based on the SVM algorithm. Return the f1_score of the prediction. """

    X_train = list(X_train)
    y_train = list(y_train)
    X_test = list(X_test)
    y_test = list(y_test)

    svc = svm.SVC(kernel='linear', C=1, probability=True)
    svc = svc.fit(X_train, y_train)
    # predict on the test set
    y_pred = svc.predict(X_test)
    # evaluate on the test set and return f1 score
    return f1_score(y_test, y_pred, average='macro')


def KNN(X_train, y_train, X_test, y_test, n_neighbors):
    """ Classify test set based on the KNN algorithm. Return the f1_score of the prediction. """

    # TRAIN TEST SPLIT
    # xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(
    #     X_train, y_train, random_state=42, test_size=0.2)  # input for this method is any array of features

    # xtrain_bow = list(xtrain_bow)
    # xvalid_bow = list(xvalid_bow)
    # ytrain = list(ytrain)
    # yvalid = list(yvalid)

    # knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    # knn = knn.fit(xtrain_bow, ytrain)
    # y_pred = knn.predict(xvalid_bow)  # predict on the test set
    # # evaluate on the test set
    # return f1_score(yvalid, y_pred, average='macro')

    # ALL DATA
    X_train = list(X_train)
    y_train = list(y_train)
    X_test = list(X_test)
    y_test = list(y_test)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn = knn.fit(X_train, y_train)
    # predict on the test set
    y_pred = knn.predict(X_test)
    # evaluate on the test set and return f1 score
    return f1_score(y_test, y_pred, average='macro')
