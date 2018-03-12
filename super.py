#!/usr/bin/env python

#!/usr/bin/env python

import glob, copy, os, pickle, sys
import numpy as np
import scipy.io.wavfile as wav
import bob.learn.em

from math import floor, ceil
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
        training_feature_batch_size = 10

        data_dirnames = glob.glob('./data/*')
        speaker_names = [dirname[7:] for dirname in data_dirnames]
        speaker_names.remove('chatter')
        speakers = {}

        total_feature = 0
        # extract features for each speaker
        print "Extracting speaker sound features"
        for spk_name in speaker_names:
            data_dirnames = glob.glob('./data/'+spk_name+'/*.wav')
            speakers.update({spk_name: SPEAKER(spk_name, data_dirnames)})
            total_feature += speakers[spk_name].speaker_features.shape[0]
        print "Finish extracting speaker sound features"

        # training ubm using all speaker features
        print "Training UBM"

        # extract feature
        print "Chatter data: meeting8k.wav"
        (rate, sig) = wav.read('./data/chatter/meeting8k.wav')
        background_features = np.array(mfcc(sig, rate))
        # Feature normalization, mean subtraction
        background_features_mean = np.mean(background_features, axis=0)
        background_features = background_features - background_features_mean

        ubm_all = UBM(gaussian_num, background_features)
        # converting to bob style for MAP
        ubm_bob = bob.learn.em.GMMMachine(gaussian_num, background_features.shape[1])
        ubm_bob.means = ubm_all.get_means()
        ubm_bob.variances = ubm_all.get_covariances()

        # saving ubm to file
        pickle.dump(ubm_all, open('./model/SVM/ubm_all.mod', 'wb'))
        print "Finished training UBM"

        supervectors_all = np.empty(shape=[0, gaussian_num*feature_length])
        supervectors_label_all = []
        # build model for each speaker
        for i, spk_name in enumerate(speaker_names):
            # getting speaker features
            speaker_features = speakers[spk_name].speaker_features
            num_features = int(ceil(speaker_features.shape[0]/training_feature_batch_size))
            supervectors = np.zeros((num_features, gaussian_num*feature_length))
            supervectors_label = np.zeros(num_features)

            # build and train GMM UBM model for each speaker
            print "Getting supervector for", spk_name

            # supervectors = np.empty(shape=[0, gaussian_num*13])
            for j in range(0, num_features):
                ubm_map = UBM_MAP(ubm_bob, speaker_features[j*training_feature_batch_size:j*training_feature_batch_size+training_feature_batch_size])
                supervectors[j,:] = ubm_map.get_supervector()
                supervectors_label[j] = i
        
            supervectors_all = np.vstack((np.array(supervectors_all), supervectors))
            supervectors_label_all = np.hstack((np.array(supervectors_label_all), supervectors_label))
            print "Finished getting supervector for", spk_name

        supervectors_label_all = supervectors_label_all.astype(int)

        print "Training SVM"
        p = SVM(supervectors_all, supervectors_label_all, speaker_names)
        
        # saving trained SVM to file
        pickle.dump(p, open('./model/SVM/SVM.mod', 'wb'))
        print "Finished training SVM"

    if str(sys.argv[1]) == "run":
        # read saved SVM
        print "Fetching SVM"
        svm = pickle.load(open('./model/SVM/SVM.mod', 'rb'))

        # read saved UBM
        print "Fetching UBM"
        ubm_all = pickle.load(open('./model/SVM/ubm_all.mod', 'rb'))
        # converting to bob style for MAP
        ubm_bob = bob.learn.em.GMMMachine(ubm_all.num_gauss, 13)
        ubm_bob.means = ubm_all.get_means()
        ubm_bob.variances = ubm_all.get_covariances()

        # extract feature
        print "Test data:", str(sys.argv[2])
        (rate, sig) = wav.read(str(sys.argv[2]))
        mfcc_feat = np.array(mfcc(sig, rate))
        # Feature normalization, mean subtraction
        mean = np.mean(mfcc_feat, axis=0)
        mfcc_feat = mfcc_feat - mean

        # get supervector
        ubm_map = UBM_MAP(ubm_bob, mfcc_feat)
        test_supervectors = np.array(ubm_map.get_supervector(), dtype='float64')

        print "Predicted speaker is", svm.predict(test_supervectors)