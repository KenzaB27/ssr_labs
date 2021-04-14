# ssr_labs
Labs of DD2119 Speech and Speaker Recognition @ KTH

# DD2119 Speech and Speaker Recognition course
Author: *Anna Sánchez Espunyes,  Kenza Bouzid*

Solutions for labs for SSR course @ KTH. Each lab contains implementation of speech recognition algorithms as well as notebooks with experiments.

## Lab1 - Feature Extraction

The objective is to experiment with different features commonly used for speech analysis and recognition. 

#### Tasks:

* compute Mel Filterbank and MFCC features step-by-step
* examine features
*  evaluate correlation between feature
*  compare utterances with Dynamic Time Warping
*  illustrate the discriminative power of the features with respect to words
*  perform hierarchical clustering of utterances
*  train and analyze a Gaussian Mixture Model of the feature vectors. 


## Lab2 - Hidden Markov Models with Gaussian Emissions

#### Objectives:

* implement the algorithms for the evaluation and decoding of Hidden Markov Models
  (HMMs),
* use your implementation to perform isolated word recognition
*  implement the algorithms for training Gaussian Hidden Markov Models (G-HMMs),
*  explain the meaning of the forward, backward and state posterior probabilities evaluated
  on speech utterances,

#### Tasks:

The overall task is to implement and test methods for isolated word recognition:

* combine phonetic HMMs into word HMMs using a lexicon
*  implement the forward-backward algorithm,
*  use it compute the log likelihood of spoken utterances given a  Gaussian HMM
*  perform isolated word recognition
*  implement the Viterbi algorithm, and use it to compute Viterbi path and likelihood
*  compare and comment Viterbi and Forward likelihoods
*  implement the Baum-Welch algorithm to update the parameters of the emission probability
  distributions

## Lab3 - Phoneme Recognition with Deep Neural Networks

#### Objectives:

* create phonetic annotations of speech recordings using predefined phonetic models
*  use software libraries1 to define and train Deep Neural Networks (DNNs) for phoneme recognition
*  explain the difference between HMM and DNN training
* compare which speech features are more suitable for each model and explain why

#### Tasks:

* using predefined Gaussian-emission HMM phonetic models, create time aligned phonetic
  transcriptions of the TIDIGITS database,
* define appropriate DNN models for phoneme recognition using Keras,
* train and evaluate the DNN models on a frame-by-frame recognition score,
* repeat the training by varying model parameters and input features