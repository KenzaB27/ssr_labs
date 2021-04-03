# DT2119, Lab 1 Feature Extraction
import numpy as np

import scipy.signal as signal
import scipy.fftpack as fftpack
from lab1_tools import lifter, trfbank
import matplotlib.pyplot as plt
# Function given by the exercise ----------------------------------

def mspec(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, samplingrate=20000):
    """Computes Mel Filterbank features.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        samplingrate: sampling rate of the original signal

    Returns:
        N x nfilters array with mel filterbank features (see trfbank for nfilters)
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)

def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    mspecs = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)
    ceps = cepstrum(mspecs, nceps)
    return lifter(ceps, liftercoeff)

# Functions to be implemented ----------------------------------

def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """
    n = samples.shape[0]

    slices = n // winlen * 2 - 1
    frames = np.zeros((slices, winlen))
    for i in range(slices):
        frames[i] = samples[i*winshift:i*winshift+winlen]

    return frames
    
def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """
    a = 1
    b = np.array([1, -p])
    N, M = input.shape

    pre_emp = np.zeros((N, M))
    for i in range(N):
        pre_emp[i] = signal.lfilter(b, a, input[i])

    return pre_emp
    
def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """

    N, M = input.shape
    windowed_samples = np.zeros((N, M))
    window = signal.hamming(M, sym=False)

    for i in range(N):
        windowed_samples[i] = input[i] * window

    return windowed_samples

def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """

    _fft = fftpack.fft(input,nfft)
    modulus = abs(_fft)
    mod_squared = modulus ** 2
    
    return mod_squared

def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in lab1_tools.py to calculate the filterbank shapes and
          nmelfilters
    """

    N, nfft = input.shape  # 92frames
    Mel = trfbank(samplingrate, nfft)  # 40filters*512
    M = Mel.shape[0]

    logMelSpec = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            logMelSpec[i, j] = np.log(np.sum(input[i]*Mel[j]))

    return logMelSpec

def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """

    cepstrum = fftpack.dct(input)[:,:nceps]
    # cepstrum = fftpack.dct(input, n=nceps)

    return cepstrum

def dtw(x, y, dist):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """
    N, M = x.shape[0], y.shape[0]
    LD = np.zeros((N, M))
    AD = np.zeros((N, M))
    predecessors = {(i,j):[0,0] for i in range(N) for j in range(M)}
    
    for i in range(N):
        for j in range(M):
            LD[i, j] = dist(x[i]-y[j])

    AD[0,0] = LD[0,0]

    for i in range(1, N):
        AD[i,0] = LD[i,0] + AD[i-1,0]

    for i in range(1, M):
        AD[0,i] = LD[0,i] + AD[0,i-1]

    for i in range(1,N):
        for j in range(1,M):
            poss = [AD[i-1, j], AD[i-1, j-1], AD[i, j-1]]
            predecessors[(i,j)] = [i-1, j] if np.argmin(poss) == 0 else ([i-1, j-1] if np.argmin(poss) == 1 else [i, j-1])
            AD[i, j] = LD[i, j] + min(AD[i-1,j], AD[i-1, j-1], AD[i, j-1])

    # Backtrack
    node = (N-1, M-1)
    path = [[N-1, M-1]]
    while node != (0,0):
        path.append(predecessors[node])
        node = tuple(predecessors[node])
    path.reverse()
    path = np.array(path)

    d = AD[N-1, M-1] / (N+M)

    return d, LD, AD, path





