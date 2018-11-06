
## Module train


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
    