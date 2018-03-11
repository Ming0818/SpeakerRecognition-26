#!/usr/bin/env python

import numpy as np
import scipy.io.wavfile as wav
import bob.learn.em
import bob.learn.libsvm

from sklearn.mixture import GaussianMixture
from python_speech_features import mfcc

class SPEAKER(object):
    def __init__(self, name, data_directory):
        self.speaker_name = name
        self.speaker_dataDirectory = data_directory
        self.speaker_features = self.extract_feature()
        self.GMM_UBM = None

    def extract_feature(self):
        (rate, sig) = wav.read(self.speaker_dataDirectory)
        mfcc_feat = np.array(mfcc(sig, rate))

        # Feature normalization, mean subtraction
        mean = np.mean(mfcc_feat, axis=0)
        mfcc_feat = mfcc_feat - mean
        
        return mfcc_feat

class GMM_UBM(object):
    def __init__(self, num_comp, spk, bck):
        self.GMM = GaussianMixture(n_components=num_comp)
        self.UBM = GaussianMixture(n_components=num_comp)
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
        self.UBM = bob.learn.em.GMMMachine(num_comp, bck.shape[1])
        self.UBM_Trainer = bob.learn.em.ML_GMMTrainer(True, True, True)
        bob.learn.em.train(self.UBM_Trainer, self.UBM, bck, max_iterations=200, convergence_threshold=1e-5)

    def get_mean(self):
        return self.UBM.means


class UBM_MAP(object):
    def __init__(self, ubm, spk):
        self.MAP = bob.learn.em.GMMMachine(ubm.shape[1], spk.shape[1])
        self.MAP_Trainer = bob.learn.em.MAP_GMMTrainer(ubm, relevance_factor=4)
        bob.learn.em.train(self.MAP_Trainer, self.MAP, spk, max_iteration=200, convergence_threshold=1e-5)

    def get_mean(self):
        return self.MAP.means

    def get_supervector(self):
        return self.MAP.mean_supervector

class SVM(object):
    def __init__(self, training_data, data_label):
        self.trainer = bob.learn.libsvm.Trainer()
        self.machine = self.trainer.train(training_data)
        self.label = data_label

    def predict(self, test_data):
        label_index = self.machine(test_data)
        return self.label[label_index]

class PREDICTOR(object):
    def __init__(self, spkrs):
        self.speakers = spkrs

    def predict(self, data):
        # data here is sound mfcc feature
        scores = []
        for spkr in self.speakers:
            score = spkr.GMM_UBM.get_score(data)
            scores.append((spkr.speaker_name, score))

        data_type= [('name','S10'),('score','float')]
        scores = np.array(scores, dtype=data_type)
        best_guess = np.sort(scores, order='score')[-1][0]
        print scores
        return best_guess
