from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def SVM(X_train, y_train, X_test, y_test):
    """ Classify test set based on the SVM algorithm. Return the f1 score of the prediction. """

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
    """ Classify test set based on the KNN algorithm. Return the f1 score of the prediction. """

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


def RoundRobin(X_train, y_train, X_test, y_test, n_neighbors):
    """ Classify test set based on the Round Robin classification algorithm. Return the f1 score of the prediction. KNN will be used as the classifier. For more information on Round Robin Classification read this: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.14.5074&rep=rep1&type=pdf """

    X_train = list(X_train)
    y_train = list(y_train)
    X_test = list(X_test)
    y_test = list(y_test)
    # make 3 new sublists: one with positive, one with negative and one with neutral tweets (and 3 sublists with the corresponding vectors)
    X_pos_train = []
    y_pos_train = []
    X_neg_train = []
    y_neg_train = []
    X_neu_train = []
    y_neu_train = []
    for i, a in enumerate(y_train):
        if a == 'positive':
            X_pos_train.append(X_train[i])
            y_pos_train.append(y_train[i])
        elif a == 'negative':
            X_neg_train.append(X_train[i])
            y_neg_train.append(y_train[i])
        elif a == 'neutral':
            X_neu_train.append(X_train[i])
            y_neu_train.append(y_train[i])

    knn1 = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn2 = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn3 = KNeighborsClassifier(n_neighbors=n_neighbors)

    # train first knn classifier only on positive and negative tweets
    X_posneg_train = X_pos_train + X_neg_train
    y_posneg_train = y_pos_train + y_neg_train
    knn1 = knn1.fit(X_posneg_train, y_posneg_train)
    # predict on all tweets of train set
    y1_train_pred = knn1.predict_proba(X_train)
    # predict on all tweets of test set
    y1_test_pred = knn1.predict_proba(X_test)

    # train second knn classifier only on positive and neutral tweets
    X_posneu_train = X_pos_train + X_neu_train
    y_posneu_train = y_pos_train + y_neu_train
    knn2 = knn2.fit(X_posneu_train, y_posneu_train)
    # predict on all tweets of train set
    y2_train_pred = knn2.predict_proba(X_train)
    # predict on all tweets of test set
    y2_test_pred = knn2.predict_proba(X_test)

    # train third knn classifier only on negative and neutral tweets
    X_negneu_train = X_neg_train + X_neu_train
    y_negneu_train = y_neg_train + y_neu_train
    knn3 = knn3.fit(X_negneu_train, y_negneu_train)
    # predict on all tweets of train set
    y3_train_pred = knn3.predict_proba(X_train)
    # predict on all tweets of test set
    y3_test_pred = knn3.predict_proba(X_test)

    # create a vector for each tweet with 6 features (the previous predict proba results) for both train and test tweets
    new_X_train = []
    for i in range(len(X_train)):
        new_X_train.append(y1_train_pred[i].tolist(
        ) + y2_train_pred[i].tolist() + y3_train_pred[i].tolist())
    new_X_test = []
    for i in range(len(X_test)):
        new_X_test.append(y1_test_pred[i].tolist(
        ) + y2_test_pred[i].tolist() + y3_test_pred[i].tolist())

    # train last knn classifier with new vectors created in train
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn = knn.fit(new_X_train, y_train)
    # predict on the test set (with new vectors)
    y_pred = knn.predict(new_X_test)

    # evaluate on the test set and return f1 score
    f1 = f1_score(y_test, y_pred, average='macro')

    return f1
