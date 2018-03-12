#!/usr/bin/env python

import numpy as np
import scipy.io.wavfile as wav
import scipy.stats as spstat

import bob.learn.em
import bob.learn.libsvm

from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture
from python_speech_features import mfcc

class SPEAKER(object):
    def __init__(self, name, data_directory):
        self.speaker_name = name
        self.speaker_dataDirectory = data_directory
        self.speaker_features = self.extract_feature()
        self.GMM_UBM = None

    def extract_feature(self):
        mfcc_feat_stack = np.empty(shape=[0,13])

        for filename in self.speaker_dataDirectory:
            print "Extracting feature from", filename
            (rate, sig) = wav.read(filename)
            mfcc_feat = np.array(mfcc(sig, rate))

            # Feature normalization, mean subtraction
            mean = np.mean(mfcc_feat, axis=0)
            mfcc_feat = mfcc_feat - mean
            mfcc_feat_stack = np.append(mfcc_feat_stack, mfcc_feat, axis=0)

        return mfcc_feat_stack

class GMM_UBM(object):
    def __init__(self, num_comp, spk, bck):
        self.GMM = GaussianMixture(n_components=num_comp, covariance_type='diag')
        self.UBM = GaussianMixture(n_components=num_comp, covariance_type='diag')
        self.train(spk, bck)

    def train(self, speaker_data, background_data):
        self.GMM.fit(speaker_data)
        self.UBM.fit(background_data)

    def get_score(self, data):
        # data is mfcc features extracted from voice
        scores = np.array(self.GMM.score_samples(data) - self.UBM.score_samples(data))
        socres_sum = np.clip(scores, 0, None).sum()
        return socres_sum

class UBM(object):
    def __init__(self, num_comp, bck):
        bck = np.array(bck, dtype='float64')
        self.num_gauss = num_comp
        self.ubm = GaussianMixture(n_components=num_comp, covariance_type='diag')
        self.ubm.fit(bck)

    def get_means(self):
        return self.ubm.means_

    def get_covariances(self):
        return self.ubm.covariances_

    #     self.UBM = bob.learn.em.GMMMachine(num_comp, bck.shape[1])
    #     # self.UBM_Trainer = bob.learn.em.ML_GMMTrainer(True, True, True)
    #     # bob.learn.em.train(self.UBM_Trainer, self.UBM, bck, max_iterations=200, convergence_threshold=1e-5, initialize=True)

    #     self.UBM.means = sk_ubm.means_
    #     self.UBM.variances = sk_ubm.covariances_

    # def get_means(self):
    #     return self.UBM.means

class UBM_MAP(object):
    def __init__(self, ubm, spk):
        spk = np.array(spk.reshape(-1,13), dtype='float64')
        self.MAP = bob.learn.em.GMMMachine(ubm.shape[0], spk.shape[1])
        self.MAP_Trainer = bob.learn.em.MAP_GMMTrainer(ubm, relevance_factor=4)
        bob.learn.em.train(self.MAP_Trainer, self.MAP, spk, max_iterations=200, convergence_threshold=1e-5)

    def get_means(self):
        return self.MAP.means

    def get_supervector(self):
        return self.MAP.mean_supervector

class SVM(object):
    def __init__(self, training_data, data_label, data_label_name):
        self.label_name = data_label_name
        self.svm = SVC(kernel='linear') 
        self.svm.fit(training_data, data_label)

    def predict(self, test_data):
        label_index = self.svm.predict(test_data.reshape(-1,test_data.shape[0]))
        return self.label_name[label_index[0]]

class GMM_UBM_PREDICTOR(object):
    def __init__(self, spkrs):
        self.speakers = spkrs

    def predict(self, data):
        # data here is sound mfcc feature
        scores = []
        for name in self.speakers.keys():
            score = self.speakers[name].get_score(data)
            scores.append((name, score))

        data_type= [('name','S10'),('score','float')]
        scores = np.array(scores, dtype=data_type)
        best_guess = np.sort(scores, order='score')[-1][0]
        return best_guess