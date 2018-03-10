#!/usr/bin/env python

import numpy as np
import scipy.io.wavfile as wav

from skleanr.mixture import GaussianMixture
from python_speech_features import mfcc

class SPEAKER(object):
    def __init__(self, name, data_directory):
        self.speaker_name = name
        self.speaker_dataDirectory = data_directory
        self.speaker_features = self.extract_feature()
        self.model = None

    def extract_feature(self):
            (rate, sig) = wav.read(self.speaker_dataDirectory)
            mfcc_feat = np.array(mfcc(sig, rate))

            # Feature normalization, mean subtraction
            mean = np.mean(mfcc_feat, axis=0)
            mfcc_feat = mfcc_feat - mean
            
            return mfcc_feat

class MODEL(object):
    def __init__(self, num_comp):
        self.GMM = GaussianMixture(n_components=num_comp)
        self.UBM = GaussianMixture(n_components=num_comp)

    def train(self, speaker_data, background_data):
        self.GMM.fit(speaker_data)
        self.UBM.fir(background_data)

    def score(self, data):
        # data is mfcc features extracted from voice
        scores = self.GMM(data) - self.UBM(data)
        socres_sum = np.array(scores).sum()

        return socres_sum