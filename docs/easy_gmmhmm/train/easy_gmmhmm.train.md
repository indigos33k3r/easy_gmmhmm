
## Module easy_gmmhmm.train

### get\_GMMs
 

 ```
 def get_GMMs(labels, trng_data=None, GMM_config=None, model_path='models', from_file=False)
 ```

 
 Utility function to train or load GMMHMMs based on entered

configuration and training data.

Returns a dictionary of trained GMMHMM objects. 
 
### obtain\_config
 

 ```
 def obtain_config(labels, file_path, from_file=False)
 ```

 
 Utility function to take in parameters to train individual GMMHMMs 
 
### train
 

 ```
 def train(data_path, models_path='models')
 ```

 
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
 
### wav2mfcc
 

 ```
 def wav2mfcc(labels, data_path, pickle_path='trng_data.pkl', from_file=False)
 ```

 
 Utility function to read wav files, convert them into MFCC vectors and store in a pickle file

(Pickle file is useful in case you re-train on the same data changing hyperparameters) 
