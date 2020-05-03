import scipy
import numpy as np
import matplotlib.pyplot as plt
import os
from haar import *
import progressbar
from integral_image import calc_int
from typing import List
from util import image_from, images_from_dir, PickleMixin, NumpyEncoder
from pprint import pprint, pformat
import pickle
from sklearn.feature_selection import SelectPercentile, f_classif
from numba import njit, jit
from multiprocessing import Pool
import json

'''
Define Each model type (Weak classifier, adaboost classifier, cascading clasifier)
'''

# Basic weak classifier
class WeakClassifier(PickleMixin):

    # Constructor based on a feature set, threshold and pol
    def __init__(self, feature, threshold, pol):
        self.feature = feature
        self.threshold = threshold
        self.pol = pol

    # Classify as long as our score is less than a threshold
    def classify(self, x):
        return 1 if self.pol*self.feature.score(x) < self.pol*self.threshold else 0
    
    # Training function for the weak classifier.
    def train_1d(self, x, y, weights, feature):
        '''
        x: 1 dimension data of size n samples
        y: 1 dimension labels of size n samples
        '''
        # Sum the weights when the images are positive / negative for the features
        pos_total = np.sum(weights[y==1])
        neg_total = np.sum(weights[y==0])

        # Define the minimum observable error so far as the total weights
        min_err = weights.sum()
        
        # Define counts used for calculating thresholds
        neg_seen, pos_seen = 0, 0
        pos_weights, neg_weights = 0, 0

        # Loop through all possible thresholds
        possible_thresholds = sorted(zip(x, y, weights), key=lambda key: key[0])
        for threshold, label, weight in possible_thresholds:

            # Define the influence based on past observations
            pol = 1 if neg_seen <= pos_seen else -1

            # Calculate the minimum error away from positive or negative totals:
            error = min(neg_total + pos_weights - neg_weights, pos_total + neg_weights - pos_weights)

            # If we have a new minimum error update our weak classifier values:
            if error < min_err:
                min_err = error
                self.threshold = threshold
                self.pol = pol
                self.feature = feature

            # Increment seen counts and weights:
            if label == 1:
                pos_seen += 1
                pos_weights += weight
            else:
                neg_seen += 1
                neg_weights += weight


class AdaBoostModel(PickleMixin):

    # Constructor for the AdaBoost Model taking in layering information, 
    # and defining two arrays with parameters / models
    def __init__(self, T):
        self.T = T
        self.clf = []
        self.alphas = []
    
    # Training function for the AdaBoost model.
    def train(self, pos_images, neg_images, data=None, features=None, labels=None, weights=None):
        '''
        pos_image: numpy array of numpy arrays
        neg_image: numpy array of numpy arrays
        '''
        # If called without preprocessing information, preproccess!
        if data is None or features is None and labels is None and weights is None:
            data, features, labels, weights = preprocess_data(pos_images, neg_images)
        
        # Calculate scores and labels based on the images, labels, and features
        X, y = _training_data(data, labels, features) # LABELS IS PASSED IN THEN OUT AS y
        
        # Convert features into numpy array
        features = np.asarray(features)

        # Dump out the first training data and label data
        with open("training.pkl", 'wb') as f:
            pickle.dump(X, f)
        with open("labels.pkl", 'wb') as f:
            pickle.dump(y, f)

        # Get indices for features with the highest score (10th percentile)
        indices = SelectPercentile(f_classif, percentile=10).fit(X, y).get_support(indices=True)

        # Keep only the data and features in the 10th percentile of scores
        X = X[:, indices]
        features = features[indices]

        bar = progressbar.ProgressBar()

        # For each bar in a range of the iteration value T for the classifier
        for _ in bar(range(self.T)):
            # Reduce the weights by their norms
            weights /= np.linalg.norm(weights)

            # Create weak classifiers from the current images / features
            weak_clfs = _weak_classifiers(X, y, weights, features)

            # Find the best classifier
            min_error, best_clf, results = _best_weak_classifier(weak_clfs, data, y, weights)
            results = np.array(results)

            # If we converge
            if min_error == 0 or min_error > 0.5:
                bar.finish()
                break

            # Calculate our modification factor beta
            beta = min_error/(1-min_error)
            beta_pow = np.power(beta, 1 - results)

            # Modify the weights for next iteration
            weights = weights * beta_pow

            # Create our alpha
            alpha = np.log(1.0/beta)

            # Remember the alpha in the classifier
            self.alphas.append(alpha)
            # Remember the best weak classifier in this itteration
            self.clf.append(best_clf)

        # Convert alpha and clf info into np arrays.
        self.alphas = np.asarray(self.alphas)
        self.clf = np.asarray(self.clf)
    
    # Classify images with the adaboost classifier
    def classify(self, image, recalc=True):
        # Recalculate integral image if necessary for passed image
        if recalc:
            integral_image = calc_int(image)
        else:
            integral_image = image
        
        # The adaboost classification score is the sum of best weak classifier scores multiplied by the corresponding alpha
        classify_score = np.sum([alpha * clf.classify(integral_image) for alpha, clf in zip(self.alphas, self.clf)])
        
        # Define a random threshold for classification based on the sum of alphas
        random_thresh = 0.5 * np.sum(self.alphas)

        return 1 if classify_score >= random_thresh else 0

class CascadeClassifier(PickleMixin):
    
    # Constructor for our cascade classifier. Layers input, instantiates empty array.
    def __init__(self, feature_layers):
        self.feature_layers = feature_layers
        self.clf = []

    # Function to train our model based on the input positive / negative images
    def train(self, pos_images, neg_images):
        # data, features, labels, weights = preprocess_data(pos_images, neg_images)
        
        # Temporarily store all negative images
        tmp_neg = neg_images

        # Loop through each defined layer that the model is defined with.
        for layer in self.feature_layers:
            # Create an adaboost model based on the layer value
            model = AdaBoostModel(layer)

            # Train the adaboost model on all the positive images, and remaining negative images
            model.train(pos_images, tmp_neg)

            # Retain the previously used model
            self.clf.append(model)

            # Get classification results from the model classifier and convert into boolean results
            results = [model.classify(image) for image in tmp_neg]
            results = np.asarray(results, dtype=np.bool)

            # Cut the negative results that have been classified correctly
            tmp_neg = tmp_neg[results]

            # If we ever run out of negative images, break.
            if len(tmp_neg) <= 0:
                break
    
    # Classification function for the cascade classifier:
    def classify(self, image, recalc=True):
        # For each model inside our cascade (each level)
        for clf in self.clf:
            # If any of our models classify this as false
            if clf.classify(image, recalc=recalc) == 0:
                # Return 0
                return 0
        
        # Otherwise, no issues are found and we can classify the image as true.
        return 1

# Function to take in the images and calculate relevant information / break down the data
def preprocess_data(pos_images, neg_images):
    # Create array matching image arrays with weights equal to 1 / total images (positive and negative)
    pos_weights = np.full(len(pos_images), 1/(2*len(pos_images)))
    neg_weights = np.full(len(neg_images), 1/(2*len(neg_images)))
    weights = np.concatenate((pos_weights, neg_weights), axis=0)

    # Create arrays holding labels for positive and negative images
    pos_label = np.ones_like(pos_weights)
    neg_label = np.zeros_like(neg_weights)
    labels = np.concatenate((pos_label, neg_label), axis=0)

    # Stack all the positive and negative image data
    data = np.concatenate((pos_images, neg_images), axis=0)
    
    # Loop through all the data and replace with the integral images
    for i, image in enumerate(data):
        data[i] = calc_int(image)
    
    # Calculate hte first set of base features 
    features = haar_features(data[0])

    return data, features, labels, weights

# Look at all classifiers and return the one with the least error
def _best_weak_classifier(classifiers: List[WeakClassifier], data, y, weights):
    min_error = float('inf')
    best_clf = None
    best_results = None
    
    # Loop through each weak classifier and classify each image.
    for clf in classifiers:
        results = []
        error = 0

        # Loop through each weight for the image with a certain label
        for weight, integral_image, label in zip(weights, data, y):
            # Utilize the weak classifier to classify the image
            result = clf.classify(integral_image)
            
            # Add to error based on weights
            correctness = abs(result - label)
            error += weight*correctness

            results.append(result)
        
        # Normalize the error to the length of the data
        error /= len(data)

        # Remember the beest classifier so far
        if error < min_error:
            min_error = error
            best_clf = clf
            best_results = results

    return min_error, best_clf, best_results

# Create weak classifiers for each of the features
def _weak_classifiers(X, y, weights, features):
    classifiers = []
    # Iterate through each feature and train a weak classifier based on current weights / features / labels
    for i in range(X.shape[1]):
        weak_clf = WeakClassifier(None, None, None)
        weak_clf.train_1d(X[:, i], y, weights, features[i])
        classifiers.append(weak_clf)
    return classifiers

# Compute all the scores based on the integral images and the current feature set returning the scores and labels
def _training_data(data, labels, features):
    # Create 2D array with features x data
    X = np.empty((len(data), len(features)))

    # Begin a progress bar
    bar = progressbar.ProgressBar(maxval=len(data), widgets=[progressbar.Counter('images parsed: %d')])
    bar.start()
    
    # Run through every image from the training data
    for i, integral_image in enumerate(data):
        # Update the bar:
        bar.update(i)
        
        # Look at every possible feature and generate the score for each feature x image.
        for j, feature in enumerate(features):
            score = feature.score(integral_image)
            X[i, j] = score
    
    bar.finish()
    
    return X, labels
        
if __name__ == '__main__':
    # Define the model type to be added
    # model = AdaBoostModel(10)
    
    # Define they layers for the classifiers
    layers = [2, 10, 20, 50]

    #Define number of images for each set:
    num_pos_images = 200
    num_neg_images = 200
 
    # Get all the facial images data. Option to limit number of images:
    pos_images = images_from_dir('face', limit= num_pos_images)
    neg_images = images_from_dir('nonface', limit= num_neg_images)

    # Define our cascade classifier and pass the feature layers
    model = CascadeClassifier(feature_layers=layers)

    # Run the training method on the positive images, and negative images.
    model.train(pos_images, neg_images)   

    # Print out the classification for the first face and no-face images
    print("Check face image (expected 1): ", model.classify(image_from('face/1.pgm')))
    print("Check non-face image (expected 0): ", model.classify(image_from('nonface/1.pgm')))

    # Save the computed model
    model.save('cascade_200-200')
