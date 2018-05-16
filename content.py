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

    distance_matrix = (1 - X) @ X_train.transpose()
    distance_matrix += X @ (1 - X_train.transpose())
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
    dist_sorted_args = Dist.argsort(kind='mergesort')
    return y[dist_sorted_args]


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
    # potrzebujemy tylko N1xk etykiet (są posortowane od najbliższych)
    resize_y = np.delete(y, range(k, y.shape[1]), axis=1)
    # zliczanie klas dla każdego wiersza w resized_y
    p_y_x = np.apply_along_axis(np.bincount, axis=1, arr=resize_y, minlength=classes.shape[0])
    return np.divide(p_y_x, k)


def classification_error(p_y_x, y_true):
    """
    Wyznacz blad klasyfikacji.
    :param p_y_x: macierz przewidywanych prawdopodobienstw
    :param y_true: zbior rzeczywistych etykiet klas 1xN.
    Kazdy wiersz macierzy reprezentuje rozklad p(y|x)
    :return: blad klasyfikacji
    """
    # wybierz klase k o najwyższym prawdopodobieństwie (największa klasa a)
    best_c = p_y_x.shape[1] - 1 - np.argmax(np.flip(p_y_x, axis=1), axis=1)
    # odejmij od klas prawdziwe klasy, aby sprawdzić czy są równe
    best_c = np.subtract(best_c, y_true)
    # zlicz jeśli nie są zerem (są różne)
    diff = np.count_nonzero(best_c)
    diff /= y_true.shape[0]
    return diff


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

    for k in range(len(k_values)):
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
    # zlicz wystapienia klas w wektorze ytrain
    # podziel wystąpienia przez ilość etykiet ytrain
    return np.divide(np.bincount(ytrain), ytrain.shape[0])


def estimate_p_x_y_nb(Xtrain, ytrain, a, b):
    """
    :param Xtrain: dane treningowe NxD
    :param ytrain: etykiety klas dla danych treningowych 1xN
    :param a: parametr a rozkladu Beta
    :param b: parametr b rozkladu Beta
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(x|y) zakladajac, ze x przyjmuje wartosci binarne i ze elementy
    x sa niezalezne od siebie. Funkcja zwraca macierz p_x_y o wymiarach MxD.
    """
    Xtrain = Xtrain.toarray()
    classes = np.unique(ytrain)
    rows = []
    # dla każdego wiersza m (każdej klasy)
    for m in range(classes.shape[0]):
        # zlicz k in ytrain
        y_train_k = np.equal(ytrain, m)
        y_train_k_count = sum(y_train_k)

        # zlicz jeśli k jest w ytrain i w Xtrain na pozycji n jest 1
        def sum_x_equals1_and_k(col, equals_k):
            return np.sum(np.bitwise_and(col, equals_k))

        # zlicz po kolumnach
        row = np.apply_along_axis(sum_x_equals1_and_k, axis=0, arr=Xtrain, equals_k=y_train_k)
        # podziel i dodaj do tablicy wierszy, która będzie przekonwertowana na macierz MxD
        rows.append(np.divide(np.add(row, a - 1), y_train_k_count + a + b - 2))

    # konwertuj na macierz
    return np.stack(rows)


def p_y_x_nb(p_y, p_x_1_y, X):
    """
    :param p_y: wektor prawdopodobienstw a priori o wymiarach 1xM
    :param p_x_1_y: rozklad prawdopodobienstw p(x=1|y) - macierz MxD
    :param X: dane dla ktorych beda wyznaczone prawdopodobienstwa, macierz NxD
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla kazdej z klas z wykorzystaniem klasyfikatora Naiwnego
    Bayesa. Funkcja zwraca macierz p_y_x o wymiarach NxM.
    """
    X = X.toarray()
    p_x_0_y = 1 - p_x_1_y
    p_y_x = []
    for n in range(X.shape[0]):
        success = p_x_1_y ** X[n, :]
        fail = p_x_0_y ** (1 - X)[n, :]
        temp = np.prod(success * fail, axis=1) * p_y
        # sum of p(x|y') * p(y')
        sum_down = np.sum(temp)
        p_y_x.append(temp / sum_down)
    return np.array(p_y_x)


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
    best_e_index = 0
    alen = int(len(a_values))
    blen = int(len(b_values))
    errors = []

    def find_error(index):
        nonlocal best_e_index
        i = int(index / alen)
        j = int(index % blen)

        py = estimate_a_priori_nb(ytrain)
        p_x_y = estimate_p_x_y_nb(Xtrain, ytrain, a_values[i], b_values[j])
        p_y_x = p_y_x_nb(py, p_x_y, Xval)
        error = classification_error(p_y_x, yval)
        errors.append(error)

        if errors[best_e_index] > error:
            best_e_index = index

    xx = map(find_error, range(alen * blen))
    list(xx)

    return (errors[best_e_index], a_values[int(round(best_e_index / len(b_values)))],
            b_values[best_e_index % len(b_values)], np.array(errors).reshape(len(a_values), len(b_values)))
