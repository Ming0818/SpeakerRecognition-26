#!/usr/bin/env python

import glob, copy, os, pickle, sys
import numpy as np
import scipy.io.wavfile as wav
import bob.learn.em

from python_speech_features import mfcc
from spk_rcg import *


if len(sys.argv) == 2:

    data_dirnames = glob.glob('./data/*.wav')
    speakers = {}
    speaker_names = []

    # extract features for each speaker
    print "Extracting speaker sound features"
    for dirs in data_dirnames:
        path, filename = os.path.split(dirs)
        speaker_names.append(filename[:-4])
        speakers.update({speaker_names[-1]: SPEAKER(speaker_names[-1], dirs)})
    print "Finish Extracting speaker sound features"

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

        if str(sys.argv[1]) == "run":
            # read saved model
            print "Fetching model for", spk_name
            model = pickle.load(open('./data/'+spk_name+'.mod', 'rb'))
            speakers[spk_name].GMM_UBM = model

        if str(sys.argv[1]) == "train":
            # build and train GMM UBM model for each speaker
            print "Training GMM UBM for", spk_name
            model = GMM_UBM(244, speaker_features, background_features)
            pickle.dump(model, open('./data/'+spk_name+'.mod', 'wb'))
            print "Traning GMM UBM finished for", spk_name



    print "Prediction"
    P = PREDICTOR(speakers.values())
    best_guess = P.predict(speakers['william'].speaker_features)
    print "Best Guess:", best_guess

    exit()
