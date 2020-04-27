import scipy
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from haar import *
import progressbar
from integral_image import calc_int
from typing import List
from util import image_from, images_from_dir
from pprint import pprint, pformat
import pickle
from sklearn.feature_selection import SelectPercentile, f_classif
from numba import njit, jit

LOADING_BAR_LENGTH = 50

'''
Define adaboost model + any models here
'''
class WeakClassifier:
    def __init__(self, feature, threshold, pol):
        self.feature = feature
        self.threshold = threshold
        self.pol = pol

    def classify(self, x):
        return 1 if self.pol*self.feature.score(x) < self.pol*self.threshold else 0

class AdaBoostModel:

    def __init__(self, T):
        self.T = T
        self.clf = []
        self.alphas = []
    
    def train(self, pos_images, neg_images):
        '''
        pos_image: numpy array of numpy arrays
        neg_image: numpy array of numpy arrays
        '''
        pos_weights = np.full(len(pos_images), 1/(2*len(pos_images)))
        neg_weights = np.full(len(neg_images), 1/(2*len(neg_images)))
        weights = np.concatenate((pos_weights, neg_weights), axis=0)
        pos_label = np.ones_like(pos_weights)
        neg_label = np.zeros_like(neg_weights)
        labels = np.concatenate((pos_label, neg_label), axis=0)
        data = np.concatenate((pos_images, neg_images), axis=0)
        for i, image in enumerate(data):
            data[i] = calc_int(image)
        features = haar_features(data[0])
        X, y = _training_data(data, labels, features)
        with open("training.pkl", 'wb') as f:
            pickle.dump(X, f)
        with open("labels.pkl", 'wb') as f:
            pickle.dump(y, f)
        indices = SelectPercentile(f_classif, percentile=10).fit(X, y).get_support(indices=True)
        X = X[:, indices]
        features = np.asarray(features)
        features = features[indices]
        bar = progressbar.ProgressBar()
        for _ in bar(range(self.T)):
            weights /= np.linalg.norm(weights)
            weak_clfs = _weak_classifiers(X, y, weights, features)
            min_error, best_clf, results = _best_weak_classifier(weak_clfs, data, y, weights)
            beta = min_error/(1-min_error)
            results = np.array(results)
            beta_pow = np.power(beta, 1 - results)
            weights = weights*beta_pow
            alpha = np.log(beta)
            self.alphas.append(alpha)
            self.clf.append(best_clf)
    
    def classify(self, image):
        integral_image = calc_int(image)
        classify_score = np.sum([alpha*clf.classify(integral_image) for alpha, clf in zip(self.alphas, self.clf)])
        random_thresh = 0.5*np.sum(self.alphas)
        return 1 if classify_score >= random_thresh else 0
    
    def save(self, filename):
        with open(filename+".pkl", 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename+".pkl", 'rb') as f:
            return pickle.load(f)

# @jit(forceobj=True)
def _best_weak_classifier(classifiers: List[WeakClassifier], data, y, weights):
    min_error = float('inf')
    best_clf = None
    best_results = None
    for clf in classifiers:
        results = []
        error = 0
        for weight, integral_image, label in zip(weights, data, y):
            result = clf.classify(integral_image)
            correct = abs(result - label)
            error += weight*correct
            results.append(result)
        error /= len(data)
        if error < min_error:
            min_error = error
            best_clf = clf
            best_results = results
    return min_error, best_clf, best_results

# @jit(forceobj=True)
def _weak_classifiers(X, y, weights, features):
    pos_total = np.sum(weights[y==1])
    neg_total = np.sum(weights[y==0])
    classifiers = []
    # iterate each feature
    for i in range(X.shape[1]):
        feature_vals = X[:, i]
        sorted_feature_vals = sorted(zip(weights, feature_vals, y), key=lambda x: x[1])
        pos_seen, neg_seen = 0, 0
        pos_weights, neg_weights = 0, 0
        min_error, best_feature, best_threshold, best_polarity = float('inf'), None, None, None
        for weight, feature_val, label in sorted_feature_vals:
            error = min(neg_total + pos_weights - neg_weights, pos_total + neg_weights - pos_weights)
            if error < min_error:
                min_error = error
                best_feature = features[i]
                best_threshold = feature_val
                best_polarity = 1 if neg_seen < pos_seen else -1
            if label == 1:
                pos_seen += 1
                pos_weights += weight
            else:
                neg_seen += 1
                neg_weights += weight
        classifiers.append(WeakClassifier(best_feature, best_threshold, best_polarity))
    return classifiers

# @njit
def _training_data(data, labels, features):
    X = np.empty((len(data), len(features)))
    for i, integral_image in enumerate(data):
        for j, feature in enumerate(features):
            score = feature.score(integral_image)
            X[i, j] = score
    return X, labels

if __name__ == '__main__':
    model = AdaBoostModel(10)
    pos_images = []
    limit = 20

    # neg_images = []
    # for filename in os.listdir('nonface'):
    #     neg_images.append(image_from(filename))
    
    pos_images = images_from_dir('face')
    neg_images = images_from_dir('nonface')
    model.train(pos_images, neg_images)   
    print(model.classify(image_from('face/1.pgm')))
    print(model.classify(image_from('nonface/1.pgm')))
    print(model.alphas)
    for clf in model.clf:
        print(vars(clf))
    model.save('test_run')
    # pos_weights = np.full(len(pos_images), 1/(2*len(pos_images)))
    # neg_weights = np.full(len(neg_images), 1/(2*len(neg_images)))
    # weights = np.concatenate((pos_weights, neg_weights), axis=0)
    # pos_label = np.ones_like(pos_weights)
    # neg_label = np.zeros_like(neg_weights)
    # labels = np.concatenate((pos_label, neg_label), axis=0)
    # data = np.concatenate((pos_images, neg_images), axis=0)
    # for i, image in enumerate(data):
    #     data[i] = calc_int(image)
    # features = haar_features(data[0])
    # X, y = _training_data(data, labels, features)
    # pprint(X)
    # indices = SelectPercentile(f_classif, percentile=10).fit(X, y).get_support(indices=True)
    # X = X[:, indices]
    # features = np.asarray(features)
    # features = features[indices]
    # pprint(X)
    # pprint(features)
    # pprint(pos_weights)
    # pprint(neg_weights)
    # pprint(labels)
    # pprint(len(features))