from hmmlearn.hmm import GMMHMM
import scipy.io.wavfile as wvf
from python_speech_features import mfcc
import os
import pickle
import glob
import numpy as np
from os.path import dirname


def wav2mfcc(labels, data_path, pickle_path="trng_data.pkl", from_file=False):
    """
        Utility function to read wav files, convert them into MFCC vectors and store in a pickle file
        (Pickle file is useful in case you re-train on the same data changing hyperparameters)
    """
    trng_data = {}
    if from_file and os.path.isfile(pickle_path):
        write_pickle = False
        trng_data = pickle.load(open(pickle_path, "rb"))
    else:
        write_pickle = True
        for gender in labels:
            mfccs = []
            for wavfile in glob.glob(data_path + '/' + gender + '/*.wav'):
                rate, sig = wvf.read(wavfile)
                mfcc_feat = mfcc(sig, rate)
                mfccs.append(mfcc_feat)
            trng_data[gender] = mfccs
    if write_pickle:
        pickle.dump(trng_data, open(pickle_path, "wb"))
    return trng_data


def obtain_config(labels, file_path, from_file=False):
    """
        Utility function to take in parameters to train individual GMMHMMs
    """
    conf = {}
    if not from_file:
        for label in labels:
            conf[label] = {}
            print(label)
            conf[label]["n_components"] = int(
                eval(input("Enter number of components in the GMMHMM: ")))
            conf[label]["n_mix"] = int(
                eval(
                    input("Enter number of mixtures in the Gaussian Model: ")))
        pickle.dump(conf, open(file_path, "wb"))
    else:
        conf = pickle.load(open(file_path, "rb"))
    return conf


def get_GMMs(labels, trng_data=None, GMM_config=None,
             model_path=dirname(__file__) + "/models/gmmhmm.pkl",
             from_file=False):
    """
        Utility function to train or load GMMHMMs based on entered
        configuration and training data.
        Returns a dictionary of trained GMMHMM objects.
    """
    gmms = {}
    if not from_file:
        for label in labels:
            gmms[label] = GMMHMM(
                n_components=GMM_config[label]["n_components"],
                n_mix=GMM_config[label]["n_mix"])
            if trng_data[label]:
                # print np.shape(trng_data[wav_file])
                gmms[label].fit(np.vstack(trng_data[label]))
                # emo_machines[wav_file].fit(trng_data[wav_file])
        pickle.dump(gmms, open(model_path, "wb"))
    else:
        gmms = pickle.load(open(model_path, "rb"))
    return gmms


def train(data_path,
          models_path=dirname(__file__) + "/models"):
    """
    Description:
        Function that takes in the directory containing training data as raw wavfiles within folders named according to label and extracts MFCC feature vectors from them,
        accepts a configuration for each in terms of number of states for HMM and number of mixtures in the Gaussian Model and then trains a set of GMMHMMs,
        one for each label.
    Params:
        * data_path (mandatory): Path to the training wav files. Each folder in this path is a label and must NOT be empty.
        * models (mandatory): Path to store the generated pickle files in.

    Return:
        A python dictionary of GMMHMMs that are trained, key values being
        labels extracted from folder names.
    """
    labels = os.listdir(data_path)
    trng_data = wav2mfcc(labels, data_path, pickle_path=models_path +
                                                               '/trng_data.pkl')
    GMM_config = obtain_config(labels, file_path=models_path + '/gmm_conf.pkl')
    gmms = get_GMMs(labels, trng_data, GMM_config, model_path=models_path +
                                                                 '/gmmhmm.pkl')
    return gmms



