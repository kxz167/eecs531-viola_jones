import numpy as np
import time 
from model import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def scores(clf, pos_images, neg_images):
    correct = 0
    true_negative, false_negative = 0, 0
    true_positive, false_positive = 0, 0
    classification_time = 0
    pos_label = np.ones(len(pos_images))
    neg_label = np.zeros(len(neg_images))
    labels = np.concatenate((pos_label, neg_label), axis=0)
    indices = np.arange(labels.shape[0])
    data = np.concatenate((pos_images, neg_images), axis=0)
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    preds = []
    confidences = []
    for x, y in zip(data, labels):
        start = time.time()
        prediction = clf.classify(x)
        confidences.append(model_confidence(model, x))
        preds.append(prediction)
        classification_time += time.time() - start
        if prediction == 1 and y == 0:
            false_positive += 1
        elif prediction == 0 and y == 1:
            false_negative += 1
        elif prediction == 1 and y == 1:
            true_positive += 1
        elif prediction == 0 and y == 0:
            true_negative += 1
        correct += 1 if prediction == y else 0
    tpr, fpr = np.zeros(len(confidences)+1), np.zeros(len(confidences)+1)
    tpr[0], fpr[0] = 0, 0
    bar = progressbar.ProgressBar(maxval=len(data))
    bar.start()
    count = 0
    for i, cf in enumerate(confidences):
        count += 1
        bar.update(count)
        cf_pos = confidences >= cf
        tp = np.sum((cf_pos==1)&(labels==1))
        fp = np.sum((cf_pos==1)&(labels==0))
        tn = np.sum((cf_pos==0)&(labels==0))
        fn = np.sum((cf_pos==0)&(labels==1))
        tpr[i+1] = tp/(tp + fn)
        fpr[i+1] = fp/(fp + tn)
    bar.finish()
    tpr = np.sort(tpr)
    fpr = np.sort(fpr)
    print("False Positive Rate: %d/%d (%f)" % (false_positive, len(neg_images), false_positive/len(neg_images)))
    print("False Negative Rate: %d/%d (%f)" % (false_negative, len(pos_images), false_negative/len(pos_images)))
    print("Accuracy: %d/%d (%f)" % (correct, len(data), correct/len(data)))
    print("Average Classification Time: %f" % (classification_time / len(data)))
    return fpr, tpr, preds, labels

def model_confidence(model, image):
    integral_image = np.pad(calc_int(image), ((1, 0), (1, 0)))
    classify_score = np.sum([alpha*clf.classify(integral_image) for alpha, clf in zip(model.alphas, model.clf)])
    return classify_score

if __name__ == "__main__":
    model = AdaBoostModel.load('test_run')
    pos_images = images_from_dir('face')
    neg_images = images_from_dir('nonface')
    fpr, tpr, preds, labels = scores(model, pos_images, neg_images)
    plt.plot(fpr, tpr)
    plt.plot(tpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()