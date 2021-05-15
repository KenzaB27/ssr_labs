import numpy as np
from lab3_tools2 import *


def words2phones(wordList, pronDict, addSilence=True, addShortPause=True):
    """ word2phones: converts word level to phone level transcription adding silence

    Args:
       wordList: list of word symbols
       pronDict: pronunciation dictionary. The keys correspond to words in wordList
       addSilence: if True, add initial and final silence
       addShortPause: if True, add short pause model "sp" at end of each word
    Output:
       list of phone symbols
    """

    phon_symbols = []
    for word in wordList:
        phon_symbols += pronDict[word]
        if addShortPause:
            phon_symbols.append('sp')
    if addSilence:
        phon_symbols.append('sil')
        phon_symbols.insert(0, 'sil')
    return phon_symbols


def forcedAlignment(lmfcc, phoneHMMs, phoneTrans):
    """ forcedAlignmen: aligns a phonetic transcription at the state level

    Args:
       lmfcc: NxD array of MFCC feature vectors (N vectors of dimension D)
              computed the same way as for the training of phoneHMMs
       phoneHMMs: set of phonetic Gaussian HMM models
       phoneTrans: list of phonetic symbols to be aligned including initial and
                   final silence

    Returns:
       list of strings in the form phoneme_index specifying, for each time step
       the state from phoneHMMs corresponding to the viterbi path.
    """


def hmmLoop(hmmmodels, namelist=None):
    """ Combines HMM models in a loop

    Args:
       hmmmodels: list of dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to combine, if None,
                 all the models in hmmmodels are used

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models
       stateMap: map between states in combinedhmm and states in the
                 input models.

    Examples:
       phoneLoop = hmmLoop(phoneHMMs)
       wordLoop = hmmLoop(wordHMMs, ['o', 'z', '1', '2', '3'])
    """


def acoustic_context(feature, stack_factor=3):
    # (time, coef)
    time_steps = feature.shape[0]
    l = time_steps - 1
    stacked_features = []
    for i in range(time_steps):
        if i >= stack_factor and i < time_steps - stack_factor:
            stacked_features.append(feature[i-stack_factor: i+stack_factor+1])
        elif i == 0:
            indices = [3, 2, 1, 0, 1, 2, 3]
            stacked_features.append(feature[indices])
        elif i == 1:
            indices = [2, 1, 0, 1, 2, 3, 4]
            stacked_features.append(feature[indices])
        elif i == 2:
            indices = [1, 0, 1, 2, 3, 4, 5]
            stacked_features.append(feature[indices])
        elif i == l:
            indices = [l-3, l-2, l-1, l, l-1, l-2, l-3]
            stacked_features.append(feature[indices])
        elif i == l-1:
            indices = [l-4, l-3, l-2, l-1, l, l-1, l-2]
            stacked_features.append(feature[indices])
        elif i == l-2:
            indices = [l-5, l-4, l-3, l-2, l-1, l, l-1]
            stacked_features.append(feature[indices])
    return np.array(stacked_features)
