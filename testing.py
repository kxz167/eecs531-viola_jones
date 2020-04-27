import numpy as np
import time 
from model import *

def evaluate(clf, pos_images, neg_images):
    correct = 0
    all_negatives, all_positives = 0, 0
    true_negatives, false_negatives = 0, 0
    true_positives, false_positives = 0, 0
    classification_time = 0
    pos_label = np.ones(len(pos_images))
    neg_label = np.zeros(len(neg_images))
    labels = np.concatenate((pos_label, neg_label), axis=0)
    data = np.concatenate((pos_images, neg_images), axis=0)
    for x, y in zip(data, labels):
        if y == 1:
            all_positives += 1
        else:
            all_negatives += 1

        start = time.time()
        prediction = clf.classify(x)
        classification_time += time.time() - start
        if prediction == 1 and y == 0:
            false_positives += 1
        if prediction == 0 and y == 1:
            false_negatives += 1
        
        correct += 1 if prediction == y else 0
    
    print("False Positive Rate: %d/%d (%f)" % (false_positives, all_negatives, false_positives/all_negatives))
    print("False Negative Rate: %d/%d (%f)" % (false_negatives, all_positives, false_negatives/all_positives))
    print("Accuracy: %d/%d (%f)" % (correct, len(data), correct/len(data)))
    print("Average Classification Time: %f" % (classification_time / len(data)))

if __name__ == "__main__":
    model = AdaBoostModel.load('test_run')
    pos_images = images_from_dir('face')
    neg_images = images_from_dir('nonface')
    evaluate(model, pos_images, neg_images)
