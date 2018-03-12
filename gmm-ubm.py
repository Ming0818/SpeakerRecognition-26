#!/usr/bin/env python

import glob, copy, os, pickle, sys
import numpy as np
import scipy.io.wavfile as wav

from python_speech_features import mfcc
from spk_rcg import *

if len(sys.argv) < 2 or (str(sys.argv[1]).lower() != 'train' and str(sys.argv[1]).lower() != 'run'):
    print "[USAGE] : ./super.py [OPTION]\n\n"
    print "[OPTION]: Train: Train model use files in ./data"
    print "          Run  : Use model saved ./model to predict input data"

else:
    if str(sys.argv[1]) == "train":
        feature_length = 13
        gaussian_num = 244
        data_dirnames = glob.glob('./data/*')

        speaker_names = [dirname[7:] for dirname in data_dirnames]
        speakers = {}

        # extract features for each speaker
        print "Extracting speaker sound features"
        for spk_name in speaker_names:
            data_dirnames = glob.glob('./data/'+spk_name+'/*.wav')
            speakers.update({spk_name: SPEAKER(spk_name, data_dirnames)})
        print "Finish extracting speaker sound features"

        # build model for each speaker
        for spk_name in speaker_names:
            # getting speaker features
            speaker_features = speakers[spk_name].speaker_features
            # getting background speaker features
            background_features = np.zeros([1, speaker_features.shape[1]])
            for bck_name in speaker_names:
                if bck_name != spk_name:
                    background_features = np.vstack((background_features, speakers[bck_name].speaker_features))
            background_features = background_features[1:]
            # build and train GMM UBM model for each speaker
            print "Training GMM UBM for", spk_name
            model = GMM_UBM(244, speaker_features, background_features)
            pickle.dump(model, open('./model/GMM_UBM/'+spk_name+'.mod', 'wb'))
            print "Traning GMM UBM finished for", spk_name

    elif str(sys.argv[1]) == "run":

        # read saved model
        model_dirnames = glob.glob('./model/GMM_UBM/*.mod')
        speaker_names = [dirname[16:-4] for dirname in model_dirnames]
        speakers = {}

        for spk_name in speaker_names:
            print "Fetching model for", spk_name
            model = pickle.load(open('./model/GMM_UBM/'+spk_name+'.mod', 'rb'))
            speakers.update({spk_name: model})

        # extract feature from testing file
        (rate, sig) = wav.read(str(sys.argv[2]))
        mfcc_feat = np.array(mfcc(sig, rate))
        # Feature normalization, mean subtraction
        mean = np.mean(mfcc_feat, axis=0)
        mfcc_feat = mfcc_feat - mean

        print "Prediction"
        P = GMM_UBM_PREDICTOR(speakers)
        best_guess = P.predict(mfcc_feat)
        print "Best Guess:", best_guess
