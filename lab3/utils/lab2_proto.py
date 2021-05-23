import numpy as np
from .lab2_tools import *
from tqdm import tqdm
# from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


def concatTwoHMMs(hmm1, hmm2):
    """ Concatenates 2 HMM models

    Args:
       hmm1, hmm2: two dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be different for each)

    Output
       dictionary with the same keys as the input but concatenated models:
          startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       twoHMMs = concatHMMs(phoneHMMs['sil'], phoneHMMs['ow'])

    See also: the concatenating_hmms.pdf document in the lab package
    """

    M1, D = hmm1["means"].shape
    M2, D = hmm2["means"].shape

    K = M1 + M2
    hmm = {}
    hmm["startprob"] = np.concatenate(
        (hmm1["startprob"][:-1], hmm1["startprob"][-1] * hmm2["startprob"]))

    hmm["transmat"] = np.zeros((K+1, K+1))
    # copying top left corner hmm1
    hmm["transmat"][:M1, :M1] = hmm1["transmat"][:M1, :M1]
    # copying bottom right corner hmm2
    hmm["transmat"][M1:, M1:] = hmm2["transmat"]

    last_state_mat = np.tile(
        np.array([hmm1["transmat"][:M1, -1]]).T, (1, M2+1))
    start_prob_rep = np.tile(
        np.array([hmm2["startprob"]]), (M1, 1))
    hmm["transmat"][:M1, M1:] = np.multiply(last_state_mat, start_prob_rep)
    # emission
    hmm["means"] = np.concatenate((hmm1["means"], hmm2["means"]), axis=0)
    hmm["covars"] = np.concatenate((hmm1["covars"], hmm2["covars"]), axis=0)

    return hmm
# this is already implemented, but based on concat2HMMs() above


def concatHMMs(hmmmodels, namelist):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: dictionary of models indexed by model name. 
       hmmmodels[name] is a dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models:
         startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
    concat = hmmmodels[namelist[0]]
    for idx in range(1, len(namelist)):
        concat = concatTwoHMMs(concat, hmmmodels[namelist[idx]])
    return concat


def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """


def forward(log_emlik, log_startprob, log_transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """
    N, M = log_emlik.shape
    log_alpha = np.zeros((N, M))
    # TODO ask if we can just drop the last state
    log_alpha[0] = log_startprob[:-1] + log_emlik[0]
    for n in range(1, N):
        for j in range(M):
            log_alpha[n, j] = logsumexp(
                log_alpha[n-1] + log_transmat[:-1, j]) + log_emlik[n, j]

    return log_alpha


def max_loglikelihood(data, HMMs, speakers="all", func="forward"):

    maxloglik = {}
    acc = 0
    for i in tqdm(range(data.shape[0])):
        key = data[i]['digit'] + "_" + \
            data[i]["speaker"] + "_" + data[i]["repetition"]
        for digit in HMMs.keys():
            obsloglik = log_multivariate_normal_density_diag(
                data[i]['lmfcc'], HMMs[digit]["means"], HMMs[digit]["covars"])
            if func == "forward":
                log_alpha = forward(obsloglik, np.log(
                    HMMs[digit]['startprob']), np.log(HMMs[digit]['transmat']))
                loglik = logsumexp(log_alpha[-1])
            else:
                # print("HEY")
                loglik, _ = viterbi
                (obsloglik, np.log(
                    HMMs[digit]['startprob']), np.log(HMMs[digit]['transmat']))
            if not key in maxloglik or maxloglik[key]['loglik'] < loglik:
                maxloglik[key] = {}
                maxloglik[key]['loglik'] = loglik
                maxloglik[key]['digit'] = digit
        if key[0] == maxloglik[key]['digit']:
            maxloglik[key]["prediction"] = True
            acc += 1
        else:
            maxloglik[key]["prediction"] = False
    np.save(f'data/maxloglik_{func}_{speakers}.npy', maxloglik)
    acc = round(acc * 100 / data.shape[0], 2)

    return acc


def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """
    N, M = log_emlik.shape
    log_beta = np.zeros((N, M))
    for n in range(N-2, -1, -1):
        for i in range(M):
            # print(i)
            log_beta[n, i] = logsumexp(
                log_transmat[i, :-1] + log_emlik[n+1, :] + log_beta[n+1])
    return log_beta


def viterbi(log_emlik, log_startprob, log_transmat, forceFinalState=False):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j
        forceFinalState: if True, start backtracking from the final state in
                  the model, instead of the best state at the last time step

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """
    N, M = log_emlik.shape
    viterbi_loglik_mat = np.zeros((N, M))
    B = np.zeros((N, M), dtype=int)
    # init
    viterbi_loglik_mat[0] = log_startprob[:-1] + log_emlik[0]

    for n in range(1, N):
        for j in range(M):
            temp = viterbi_loglik_mat[n-1] + log_transmat[:-1, j]
            B[n, j] = int(np.argmax(temp))
            viterbi_loglik_mat[n, j] = temp[B[n, j]] + log_emlik[n, j]

    finalstate = np.argmax(
        viterbi_loglik_mat[-1]) if not forceFinalState else 0  # TODO

    viterbi_path = [finalstate]
    viterbi_loglik = viterbi_loglik_mat[-1, finalstate]

    for t in range(N-1, -1, -1):
        viterbi_path.append(B[t, finalstate])
        finalstate = B[t, finalstate]

    viterbi_path.reverse()

    return viterbi_loglik, viterbi_path


def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """
    N, M = log_alpha.shape
    log_gamma = np.zeros((N, M))
    temp = logsumexp(log_alpha[-1])
    for n in range(N):
        log_gamma[n] = log_alpha[n] + log_beta[n] - temp

    return log_gamma


def statePosteriorsGMM(log_emlik):
    denom = logsumexp(log_emlik)
    
    gammas = log_emlik - denom

    return gammas


def updateMeanAndVar(X, log_gamma, varianceFloor=5.0):
    """ Update Gaussian parameters with diagonal covariance

    Args:
         X: NxD array of feature vectors
         log_gamma: NxM state posterior probabilities in log domain
         varianceFloor: minimum allowed variance scalar
    were N is the lenght of the observation sequence, D is the
    dimensionality of the feature vectors and M is the number of
    states in the model

    Outputs:
         means: MxD mean vectors for each state
         covars: MxD covariance (variance) vectors for each state
    """
    # TODO varainceFloor
    D = X.shape[1]
    _, M = log_gamma.shape
    means = np.zeros((M,D))
    covars = np.zeros((M, D))
    gamma = np.exp(log_gamma)
    for j in range(M):
        sum_gamma = np.sum(gamma[:, j])
        means[j] = np.sum(gamma[:, j].reshape(-1, 1) * X, axis=0) / sum_gamma
        covars[j] = np.sum(gamma[:, j].reshape(-1, 1) * (X - means[j])**2, axis=0) / \
            sum_gamma
    covars[covars < varianceFloor] = varianceFloor
    return means, covars

def EM(hmm, X, maxiters=20, threshold=1.0, plot=False):
    
    i = 0

    vloglik, prev_vloglik = 0, float('inf')
    loglik = []

    while i < maxiters and abs(vloglik - prev_vloglik) > threshold:

        ## Expectation
        obsloglik = log_multivariate_normal_density_diag(
            X, hmm["means"], hmm["covars"])
        log_alpha = forward(obsloglik, np.log(
            hmm["startprob"]), np.log(hmm["transmat"]))
        log_beta = backward(obsloglik, np.log(
            hmm["startprob"]), np.log(hmm["transmat"]))
        log_gamma = statePosteriors(log_alpha, log_beta)
        ## Compute log likelihood of data given model
        prev_vloglik = vloglik
        vloglik, _ = viterbi(obsloglik, np.log(
            hmm["startprob"]), np.log(hmm["transmat"]))
        loglik.append(vloglik)
        ## Maximization
        hmm['means'], hmm['covars'] = updateMeanAndVar(X, log_gamma)
        i += 1
    print(f"Convergence after {i} epochs towards {loglik[-1]}")
    if plot:
        plt.figure()
        plt.plot(loglik)
        plt.xlabel("Iterations")
        plt.ylabel("viterbi log likelihood")
        plt.title("likelihood vs iteration during EM updates")
        plt.show()
    return loglik


    

