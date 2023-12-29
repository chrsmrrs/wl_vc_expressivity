import math as m

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC


# Omit sklearn warnings.
def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

warnings.filterwarnings("ignore")


# l2 normalization for feature vectors.
def normalize_feature_vector_dense(feature_vectors):
    transformer = Normalizer().fit(feature_vectors)
    return transformer.transform(feature_vectors)


# Cosine normalization for a gram matrix.
def normalize_gram_matrix(gram_matrix):
    n = gram_matrix.shape[0]
    gram_matrix_norm = np.zeros([n, n], dtype=np.double)

    for i in range(0, n):
        for j in range(i, n):
            if not (gram_matrix[i][i] == 0.0 or gram_matrix[j][j] == 0.0):
                g = gram_matrix[i][j] / m.sqrt(gram_matrix[i][i] * gram_matrix[j][j])
                gram_matrix_norm[i][j] = g
                gram_matrix_norm[j][i] = g

    return gram_matrix_norm


# 10-CV for kernel svm and hyperparameter selection.
def kernel_svm_evaluation(all_matrices, classes, num_repetitions=10,
                          C=[10 ** 3, 10 ** 2, 10 ** 1, 10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3]):
    # Acc. over all repetitions.
    test_accuracies_all = []

    for i in range(num_repetitions):
        # Test acc. over all folds.
        test_accuracies = []
        kf = KFold(n_splits=10, shuffle=True)

        for train_index, test_index in kf.split(list(range(len(classes)))):
            # Determine hyperparameters
            train_index, val_index = train_test_split(train_index, test_size=0.1)
            best_val_acc = 0.0
            best_gram_matrix = all_matrices[0]
            best_c = C[0]

            for gram_matrix in all_matrices:
                train = gram_matrix[train_index, :]
                train = train[:, train_index]
                val = gram_matrix[val_index, :]
                val = val[:, train_index]

                c_train = classes[train_index]
                c_val = classes[val_index]

                for c in C:
                    clf = SVC(C=c, kernel="precomputed")
                    clf.fit(train, c_train)
                    val_acc = accuracy_score(c_val, clf.predict(val)) * 100.0

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_c = c
                        best_gram_matrix = gram_matrix

            # Determine test accuracy.
            train = best_gram_matrix[train_index, :]
            train = train[:, train_index]
            test = best_gram_matrix[test_index, :]
            test = test[:, train_index]

            c_train = classes[train_index]
            c_test = classes[test_index]
            clf = SVC(C=best_c, kernel="precomputed", tol=0.001)
            clf.fit(train, c_train)
            best_test = accuracy_score(c_test, clf.predict(test)) * 100.0

            test_accuracies.append(best_test)

        test_accuracies_all.append(float(np.array(test_accuracies).mean()))

    return (np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std())


# 10-CV for linear svm with sparse feature vectors and hyperparameter selection.
def linear_svm_evaluation(all_feature_matrices, classes, num_iter, num_repetitions=10,
                          C=[10 ** 3, 10 ** 2, 10 ** 1, 10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3]):
    # Acc. over all repetitions.
    test_accuracies_all = []

    # Margins over all repetitions.
    margins_all = []

    for i in range(num_repetitions):
        # Test acc. over all folds.
        test_accuracies = []
        # Margins over all folds.
        margins = []
        kf = KFold(n_splits=10, shuffle=True)

        for train_index, test_index in kf.split(list(range(len(classes)))):
            # Sample 10% split from training split for validation.
            train_index, val_index = train_test_split(train_index, test_size=0.1)
            best_val_acc = 0.0
            best_gram_matrix = all_feature_matrices[0]
            best_c = C[0]

            for gram_matrix in all_feature_matrices:
                train = gram_matrix[train_index]
                val = gram_matrix[val_index]

                c_train = classes[train_index]
                c_val = classes[val_index]

                for c in C:
                    clf = SVC(C=c, kernel="linear", max_iter=num_iter, cache_size=5000)
                    # clf = LinearSVC(C=c, loss="hinge")
                    clf.fit(train, c_train)
                    val_acc = accuracy_score(c_val, clf.predict(val)) * 100.0

                    if val_acc > best_val_acc:
                        # Get test acc.
                        best_val_acc = val_acc
                        best_c = c
                        best_gram_matrix = gram_matrix

            # Determine test accuracy.
            train = best_gram_matrix[train_index]
            test = best_gram_matrix[test_index]

            c_train = classes[train_index]
            c_test = classes[test_index]
            clf = SVC(C=best_c, kernel="linear", max_iter=num_iter, cache_size=5000)
            # clf = LinearSVC(C=best_c, loss="hinge")
            clf.fit(train, c_train)

            m = 1.0 / np.linalg.norm(clf.coef_)
            margins.append(m)
            # print(m)

            best_test = accuracy_score(c_test, clf.predict(test)) * 100.0

            test_accuracies.append(best_test)

        test_accuracies_all.append(float(np.array(test_accuracies).mean()))
        margins_all.append(float(np.array(margins).mean()))

    return (np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std(), np.array(margins_all).mean(),
            np.array(margins_all).std())
