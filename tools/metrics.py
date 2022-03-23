import numpy as np
from scipy import interpolate

from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

from face_dataset import LABELS

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(1 - dist, 1 - threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame),
                       np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)

    acc = float(tp + tn) / dist.shape[0]
    return tpr, fpr, acc


def calculate(threshold, dist, actual_issame):
    predict_issame = np.less(1 - dist, 1 - threshold)
    print(predict_issame.shape)
    print(actual_issame.shape)

    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame),
                       np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    prob = prob.bool()
    label = label.bool()
    epsilon = 1e-7
    tp = (prob & label).sum().float()
    tn = ((~prob) & (~label)).sum().float()
    fp = (prob & (~label)).sum().float()
    fn = ((~prob) & label).sum().float()

    return tp, fp, tn, fn


def F1_score(y_pred, y_true):
    print("\n",classification_report(y_true, y_pred,target_names=LABELS))

    precision = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
    recall = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')

    return f1, precision, recall


def TPR_FPR(dist, actual_issame, fpr_target=0.001):
    # acer_min = 1.0
    # thres_min = 0.0
    # re = []

    # Positive
    # Rate(FPR):
    # FPR = FP / (FP + TN)

    # Positive
    # Rate(TPR):
    # TPR = TP / (TP + FN)

    thresholds = np.arange(0.0, 1.0, 0.0001)
    nrof_thresholds = len(thresholds)

    fpr = np.zeros(nrof_thresholds)
    tpr = np.zeros(nrof_thresholds)

    FPR = 0.0
    TPR = 0.0
    for threshold_idx, threshold in enumerate(thresholds):

        if threshold < 1.0:
            tp, fp, tn, fn = calculate(threshold, dist, actual_issame)
            FPR = fp / (fp * 1.0 + tn * 1.0)
            TPR = tp / (tp * 1.0 + fn * 1.0)
            # print(threshold, FPR, TPR) 

        fpr[threshold_idx] = FPR
        tpr[threshold_idx] = TPR

    differ_tpr_fpr_1=tpr+fpr-1.0

    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = thresholds[right_index]
    err = fpr[right_index]   
    print(best_th, err)
    # print(np.argmax(fpr))
    # print(thresholds[np.argmax(fpr)])

    if np.max(fpr) >= fpr_target:
        f = interpolate.interp1d(np.asarray(fpr), thresholds, kind='slinear')
        threshold = f(fpr_target)
    else:
        threshold = 0.0

    tp, fp, tn, fn = calculate(threshold, dist, actual_issame)

    FPR = fp / (fp * 1.0 + tn * 1.0)
    TPR = tp / (tp * 1.0 + fn * 1.0)

    print(threshold, TPR, FPR, tp, fp, tn, fn)
    return FPR, TPR
