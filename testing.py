import numpy as np
import time 
from model import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from util import PickleMixin

def scores(clf, pos_images, neg_images):
    # Define variables for remember classification results
    correct = 0

    true_negative, false_negative = 0, 0
    true_positive, false_positive = 0, 0
    
    classification_time = 0
    
    #Create positive and negative classifications for each image
    pos_label = np.ones(len(pos_images))
    neg_label = np.zeros(len(neg_images))
    labels = np.concatenate((pos_label, neg_label), axis=0)

    # Create our data array for all the image data
    data = np.concatenate((pos_images, neg_images), axis=0)

    # Define all the indices for each label (image)
    indices = np.arange(labels.shape[0])

    # Randomize the order that we will be pulling from the indices.
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    # Empty arrays to add predictions and confidence levels
    preds = []
    confidences = []

    # Create a progress bar over all the images
    print("Classifying all images:...")
    bar = progressbar.ProgressBar(maxval=len(data))
    bar.start()
    count = 0

    # Iterate through each data - label pair with x set as data, y set as labels
    for x, y in zip(data, labels):
        count += 1
        bar.update(count)

        #Begin timer to time classification:
        start = time.time()

        # Classify image based on the model
        prediction = clf.classify(x)
        preds.append(prediction)

        # Determine the classification confidence
        confidences.append(model_confidence(model, x))

        # Add to our total classification time
        classification_time += time.time() - start

        # Increment f/t pos/neg counts
        if prediction == 1 and y == 0:
            false_positive += 1
        elif prediction == 0 and y == 1:
            false_negative += 1
        elif prediction == 1 and y == 1:
            true_positive += 1
        elif prediction == 0 and y == 0:
            true_negative += 1
        
        #Increment correct count
        correct += 1 if prediction == y else 0

    bar.finish()

    # Create a bunch of zeros of the length for total images classified
    tpr, fpr = np.zeros(len(confidences)+1), np.zeros(len(confidences)+1)
    tpr[0], fpr[0] = 0, 0
    
    # Create a progress bar over all the images
    print("Calculating confidence results:")
    bar = progressbar.ProgressBar(maxval=len(data))
    bar.start()

    # Iterate over all confidence results
    for i, cf in enumerate(confidences):
        bar.update(i)
        
        # Get all confidences which are greater than current confidence level
        cf_pos = confidences >= cf

        # Total t/f p/n
        tp = np.sum((cf_pos==1)&(labels==1))
        fp = np.sum((cf_pos==1)&(labels==0))
        tn = np.sum((cf_pos==0)&(labels==0))
        fn = np.sum((cf_pos==0)&(labels==1))

        # Calculate the rates for true positive / false positive:
        tpr[i+1] = tp/(tp + fn) if tp != 0 else 0
        fpr[i+1] = fp/(fp + tn) if fp != 0 else 0

    bar.finish()
    
    # Sort rates
    tpr = np.sort(tpr)
    fpr = np.sort(fpr)

    # Output Runing statistics
    print("False Positive Rate: %d/%d (%f)" % (false_positive, len(neg_images), false_positive/len(neg_images)))
    print("False Negative Rate: %d/%d (%f)" % (false_negative, len(pos_images), false_negative/len(pos_images)))
    print("Accuracy: %d/%d (%f)" % (correct, len(data), correct/len(data)))
    print("Average Classification Time: %f" % (classification_time / len(data)))
    return fpr, tpr, preds, labels

def model_confidence(model, image):
    # Calculate the integral image of the image to classify:
    integral_image = calc_int(image)

    if isinstance(model, AdaBoostModel):
        classify_score = np.sum([alpha*clf.classify(integral_image) for alpha, clf in zip(model.alphas, model.clf)])
        return classify_score
    else:
        for clf in model.clf:
            if clf.classify(image) == 0:
                return np.sum([alpha*weak_clf.classify(integral_image) for alpha, weak_clf in zip(clf.alphas, clf.clf)])
        return np.sum([alpha*clf.classify(integral_image) for alpha, clf in zip(model.clf[-1].alphas, model.clf[-1].clf)])

if __name__ == "__main__":
    # Utilize the load method for any classifier to run test data
    model = PickleMixin.load('cascade_50-100')

    # Get all the positive and negative images from the image directories:
    pos_images = images_from_dir('face')
    neg_images = images_from_dir('nonface')

    # Calculate all the image classifications and rates
    fpr, tpr, preds, labels = scores(model, pos_images, neg_images)
    
    # Display graph representation of the ROC curve:
    fpr2, tpr2, thr = roc_curve(labels, preds)
    plt.plot(fpr, tpr)
    plt.plot(tpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()