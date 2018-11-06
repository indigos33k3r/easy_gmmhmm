import heapq
import pickle

import scipy.io.wavfile as wvf
from python_speech_features import mfcc


def test_file(test_file, gmms):
    """
        Test a given file and predict an label for it.
    """
    rate, sig = wvf.read(test_file)
    mfcc_feat = mfcc(sig, rate)
    pred = {}
    for model in gmms:
        pred[model] = gmms[model].score(mfcc_feat)
    return get_nbest(pred, 2), pred


def get_nbest(d, n):
    """
        Utility function to return n best predictions.
    """
    return heapq.nlargest(n, d, key=lambda k: d[k])


def predict_label(test_file, model_path="models/gmmhmm.pkl"):
    """
        Description:
            predict label for input wav file.

        Params:
            * test_file (mandatory): Wav file for which label should be predicted.
            * model_path: Path to gmmhmm model.

        Return:
            A list of predicted label and next best predicted label.
    """
    gmms = pickle.load(open(model_path, "rb"))
    predicted = test_file(test_file, gmms)
    return predicted


if __name__ == "__main__":
    import time

    start = time.time()
    wav_file = "test.wav"
    predicted, probs = predict_label(wav_file)
    print("PREDICTED: %s" % predicted[0])
    print("scores: {}".format(probs))
    print("elapsed", time.time() - start)
