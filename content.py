# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

from __future__ import division

import numpy as np


def hamming_distance(X, X_train):
    """
    :param X: zbior porownwanych obiektow N1xD
    :param X_train: zbior obiektow do ktorych porownujemy N2xD
    Funkcja wyznacza odleglosci Hamminga obiektow ze zbioru X od
    obiektow X_train. Odleglosci obiektow z jednego i drugiego
    zbioru zwrocone zostana w postaci macierzy
    :return: macierz odleglosci pomiedzy obiektami z X i X_train N1xN2
    """
    X = X.toarray()
    X_train = X_train.toarray()

    distance_matrix = np.zeros(shape=(X.shape[0], X_train.shape[0]))
    for n1 in range(X.shape[0]):
        for n2 in range(X_train.shape[0]):
            for d in range(X.shape[1]):
                if X[n1][d] != X_train[n2][d]:
                    distance_matrix[n1][n2] = distance_matrix[n1][n2] + 1
    return distance_matrix


def sort_train_labels_knn(Dist, y):
    """
    Funkcja sortujaca etykiety klas danych treningowych y
    wzgledem prawdopodobienstw zawartych w macierzy Dist.
    Funkcja zwraca macierz o wymiarach N1xN2. W kazdym
    wierszu maja byc posortowane etykiety klas z y wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist
    :param Dist: macierz odleglosci pomiedzy obiektami z X
    i X_train N1xN2
    :param y: wektor etykiet o dlugosci N2
    :return: macierz etykiet klas posortowana wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist. Uzyc algorytmu mergesort.
    """
    sorted_y = np.zeros(shape=(Dist.shape[0], Dist.shape[1]))
    for n1 in range(Dist.shape[0]):
        sorted_args = Dist[n1].argsort(kind='merge')
        for i in range(sorted_args.shape[0]):
            sorted_y[n1][i] = y[sorted_args[i]]
    return sorted_y


def p_y_x_knn(y, k):
    """
    Funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla
    kazdej z klas dla obiektow ze zbioru testowego wykorzystujac
    klasfikator KNN wyuczony na danych trenningowych
    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najblizszuch sasiadow dla KNN
    :return: macierz prawdopodobienstw dla obiektow z X
    """
    classes = np.unique(y)
    p_y_x = np.zeros(shape=(y.shape[0], classes.shape[0]))
    for row in range(y.shape[0]):
        for c in range(classes.shape[0]):
            sum_index = 0
            for col in range(k):
                if classes[c] == y[row][col]:
                    sum_index = sum_index + 1
            p_y_x[row][c] = sum_index/k
    return p_y_x


def classification_error(p_y_x, y_true):
    """
    Wyznacz blad klasyfikacji.
    :param p_y_x: macierz przewidywanych prawdopodobienstw
    :param y_true: zbior rzeczywistych etykiet klas 1xN.
    Kazdy wiersz macierzy reprezentuje rozklad p(y|x)
    :return: blad klasyfikacji
    """
    error_val = 0
    for row in range(p_y_x.shape[0]):
        act_prob = 0
        act_class = p_y_x[row][0]
        for c in range(p_y_x.shape[1]):
            if p_y_x[row][c] >= act_prob:
                act_prob = p_y_x[row][c]
                act_class = c
        if act_class != y_true[row]:
            error_val = error_val + 1
    return error_val/p_y_x.shape[0]
    # p_y_x = np.fliplr(p_y_x)
    # y_truea = p_y_x.shape[1] - np.argmax(p_y_x, axis=1)
    # y_truea = np.subtract(y_truea, y_true)
    # diff = np.count_nonzero(y_truea)
    # diff /= y_true.shape[0]
    #
    # return diff


def model_selection_knn(Xval, Xtrain, yval, ytrain, k_values):
    """
    :param Xval: zbior danych walidacyjnych N1xD
    :param Xtrain: zbior danych treningowych N2xD
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartosci parametru k, ktore maja zostac sprawdzone
    :return: funkcja wykonuje selekcje modelu knn i zwraca krotke (best_error,best_k,errors), gdzie best_error to najnizszy
    osiagniety blad, best_k - k dla ktorego blad byl najnizszy, errors - lista wartosci bledow dla kolejnych k z k_values
    """
    best_k_index = 0
    errors = []
    sorted_train = sort_train_labels_knn(hamming_distance(Xval, Xtrain), ytrain)

    for k in range(k_values.shape[0]):
        error = classification_error(p_y_x_knn(sorted_train, k_values[k]), yval)
        errors.append(error)
        if errors[best_k_index] > error:
            best_k_index = k

    return errors[best_k_index], k_values[best_k_index], errors


def estimate_a_priori_nb(ytrain):
    """
    :param ytrain: etykiety dla dla danych treningowych 1xN
    :return: funkcja wyznacza rozklad a priori p(y) i zwraca p_y - wektor prawdopodobienstw a priori 1xM
    """
    classes = np.unique(ytrain)
    p_y = []
    for c in range(classes.shape[0]):
        sum_index = 0
        for col in range(ytrain.shape[0]):
            if classes[c] == ytrain[col]:
                sum_index = sum_index + 1
        p_y.append(sum_index/ytrain.shape[0])
    return p_y

    # NUMBER_OF_CLASSES = 4
    # return np.divide(np.delete(np.bincount(ytrain, minlength=NUMBER_OF_CLASSES + 1), 0), ytrain.shape[0])


def estimate_p_x_y_nb(Xtrain, ytrain, a, b):
    """
    :param Xtrain: dane treningowe NxD
    :param ytrain: etykiety klas dla danych treningowych 1xN
    :param a: parametr a rozkladu Beta
    :param b: parametr b rozkladu Beta
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(x|y) zakladajac, ze x przyjmuje wartosci binarne i ze elementy
    x sa niezalezne od siebie. Funkcja zwraca macierz p_x_y o wymiarach MxD.
    """
    pass


def p_y_x_nb(p_y, p_x_1_y, X):
    """
    :param p_y: wektor prawdopodobienstw a priori o wymiarach 1xM
    :param p_x_1_y: rozklad prawdopodobienstw p(x=1|y) - macierz MxD
    :param X: dane dla ktorych beda wyznaczone prawdopodobienstwa, macierz NxD
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla kazdej z klas z wykorzystaniem klasyfikatora Naiwnego
    Bayesa. Funkcja zwraca macierz p_y_x o wymiarach NxM.
    """
    pass


def model_selection_nb(Xtrain, Xval, ytrain, yval, a_values, b_values):
    """
    :param Xtrain: zbior danych treningowych N2xD
    :param Xval: zbior danych walidacyjnych N1xD
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param a_values: lista parametrow a do sprawdzenia
    :param b_values: lista parametrow b do sprawdzenia
    :return: funkcja wykonuje selekcje modelu Naive Bayes - wybiera najlepsze wartosci parametrow a i b. Funkcja zwraca
    krotke (error_best, best_a, best_b, errors) gdzie best_error to najnizszy
    osiagniety blad, best_a - a dla ktorego blad byl najnizszy, best_b - b dla ktorego blad byl najnizszy,
    errors - macierz wartosci bledow dla wszystkich par (a,b)
    """
    pass

