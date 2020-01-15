import math
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from numba import njit
from numpy.core.umath_tests import inner1d
from hausdorff import hausdorff_distance as hd

@njit
def confusion_matrix(segmented, ground_truth):
    
    # # Initialize the values of confusion matrix
    tp = tn = fp = fn = 0

    # Compute the values from confusion matrix
    for col in range(segmented.shape[0]):
        for row in range(segmented.shape[1]):
            if np.logical_and(segmented[col, row] == 255, ground_truth[col, row] == 255):
                tp += 1
            elif np.logical_and(segmented[col, row] == 0, ground_truth[col, row] == 0):
                tn += 1
            elif np.logical_and(segmented[col, row] == 255, ground_truth[col, row] == 0):
                fp += 1
            elif np.logical_and(segmented[col, row] == 0, ground_truth[col, row] == 255):
                fn += 1

    return tp, tn, fp, fn

class Metricas(object):
    
    iterador           = 0
    array_list_metrics = 0

    metrics_list       = []
    
    list_acc           = []
    list_pre           = []
    list_dsc           = []
    # list_hd            = []
    list_jac           = []
    list_mcc           = []
    list_sen           = []
    list_spc           = []
    list_tmp           = []

    num_img = 0
    exam = 0

    # def __init__(self, seg, doc, num_img, time):
    #
    #     self.tmp      = time
    #     self.qnt_imgs = num_img
    #     self.calc_metrics(seg, doc)

    def __init__(self, seg, doc):

        # self.tmp      = time
        # self.qnt_imgs = num_img
        self.calc_metrics(seg, doc)

    def calc_metrics(self, segmented, ground_truth):

        tp, tn, fp, fn = confusion_matrix(segmented, ground_truth)

        self.list_acc.append((tp + tn)/(tp + tn + fp + fn))
        self.list_acc = np.array(self.list_acc)

        self.list_pre.append(tp/(tp + fp))
        self.list_pre = np.array(self.list_pre)

        self.list_mcc.append((tp * tn - fp * fn)/(((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))**0.5))
        self.list_mcc = np.array(self.list_mcc)

        self.list_dsc.append(2*tp/(2*tp + fp + fn))
        self.list_dsc = np.array(self.list_dsc)

        self.list_spc.append(tn/(tn + fp))
        self.list_spc = np.array(self.list_spc)

        self.list_sen.append(tp/(tp + fn))
        self.list_sen = np.array(self.list_sen)

        self.list_jac.append(tp/(fp + fn + tp))
        self.list_jac = np.array(self.list_jac)

        # self.list_hd.append(self._hd(segmented, ground_truth))
        # self.list_hd = np.array(self.list_hd)

        # self.list_tmp.append(self.tmp)
        # self.list_tmp = np.array(self.list_tmp)

    # def _hd(self, A,B):
    #
    #     # Find pairwise distance
    #     D_mat = np.sqrt(inner1d(A,A)[np.newaxis].T + inner1d(B,B)-2*(np.dot(A,B.T)))
    #
    #     # Find DH
    #     dH = np.max(np.array([np.max(np.min(D_mat,axis=0)), np.max(np.min(D_mat,axis=1))]))
    #
    #     return(dH)/1000.0

    def save_metrics(self, num_imagem):
        if num_imagem == 1:
            file = open('results/imgs.txt', 'w')
        else:
            file = open('results/imgs.txt', 'a')
        file.write('\tImage {} \n'.format(num_imagem))
        file.write('Accuracy score           = {:.4f}\n'.format(self.list_acc[num_imagem-1]))
        file.write('Precision score          = {:.4f}\n'.format(self.list_pre[num_imagem-1]))
        file.write('Dice coefficient         = {:.4f}\n'.format(self.list_dsc[num_imagem-1]))
        file.write('Jaccard coefficient      = {:.4f}\n'.format(self.list_jac[num_imagem-1]))
        file.write('Matthews coefficient     = {:.4f}\n'.format(self.list_mcc[num_imagem-1]))
        file.write('Sensitivity coefficient  = {:.4f}\n'.format(self.list_sen[num_imagem-1]))
        file.write('Specifity coefficient    = {:.4f}\n'.format(self.list_spc[num_imagem-1]))
        # file.write('Hausdorff distance       = {:.4f}\n\n'.format(self.list_hd[num_imagem-1]))
        # file.write('Time                     = {:.4f}\n'.format(self.list_tmp[num_imagem-1]))

    def save_mean_metrics(self, n_img, exams):
        # self.array_list_metrics = (self.array_list_metrics) / self.qnt_imgs
        self.array_list_metrics = (self.array_list_metrics) / n_img

        file = open('results/general_result.txt', 'w+')
        file.write('Mean and Std Deviation \n\n')
        file.write('Accuracy score           = {:.4f} +/- {:.4f}\n'.format(np.mean(self.list_acc), np.std(self.list_acc)))
        file.write('Precision score          = {:.4f} +/- {:.4f}\n'.format(np.mean(self.list_pre), np.std(self.list_pre)))
        file.write('Dice coefficient         = {:.4f} +/- {:.4f}\n'.format(np.mean(self.list_dsc), np.std(self.list_dsc)))
        file.write('Jaccard coefficient      = {:.4f} +/- {:.4f}\n'.format(np.mean(self.list_jac), np.std(self.list_jac)))
        file.write('Matthews coefficient     = {:.4f} +/- {:.4f}\n'.format(np.mean(self.list_mcc), np.std(self.list_mcc)))
        file.write('Sensitivity coefficient  = {:.4f} +/- {:.4f}\n'.format(np.mean(self.list_sen), np.std(self.list_sen)))
        file.write('Specifity coefficient    = {:.4f} +/- {:.4f}\n'.format(np.mean(self.list_spc), np.std(self.list_spc)))
        # file.write('Hausdorff distance       = {:.4f} +/- {:.4f}\n'.format(np.mean(self.list_hd ), np.std(self.list_hd )))
        # file.write('Time                     = {:.4f} +/- {:.4f}\n'.format(np.mean(self.list_tmp), np.std(self.list_tmp)))
        file.close()
