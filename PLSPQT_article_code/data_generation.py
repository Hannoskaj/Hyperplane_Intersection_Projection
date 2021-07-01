#!/usr/bin/env python
# coding: utf-8
"""
Functions to generate sampling data from a channel and a measurement strategy.
The final result is directly the least-square estimator, the individual measurement
results are not saved.
All sampling is poissonized, that is, all possible measurement configurations are
chosen at random, a number of times independent of each other.

Main functions are:
* Choi_LS_Pauli_from_channel_mem : generates from a Pauli measurement scenario without ancilla. Close attention paid to the memory consumption. Starts directly 
from the channel.
* Choi_LS_MUBS_from_freq : generates from a MUBs measurement setting without 
ancilla
* Choi_LS_from_Pauli_freq : faster, more memory-heavy generation in a Pauli
measurement setting.

Intermediate functions include:
* prod_pauli_vecs : Generates Pauli bases for multiple qubits.
* MUBS : generates mutually unbiased bases.
* probas_Pauli : yields the probabilities of each measurement outcome in a 
Pauli setting.
* probas_MUBS : yields the probabilities of each measurement outcome in a 
MUBs setting.
"""


import numpy as N
import scipy.linalg as SL
import scipy.stats as SS
import scipy.sparse as SP
import numexpr



def prod_pauli_vecs(k, U2=None):
    """
    Outputs all the k-tensor products of Pauli vectors, as an array where the
    vectors are the lines.
    Works till k=8 included. Needs $12^k$ complex entries.
    U2 allows to add a rotation to the Pauli vectors, so as to avoid very special cases.
    """
    s2 = N.sqrt(.5)
    frame_vecs = N.array(([1,0], [0,1], [s2, s2], [s2, -s2], [s2, s2 * 1j], [s2, -s2 *1j]))
    if U2 is not None:
        frame_vecs = N.dot(frame_vecs, U2)
    einstein_indices = ('ai -> ai',
                        'ai, bj -> abij',
                        'ai, bj, ck -> abcijk',
                        'ai, bj, ck, dl -> abcdijkl',
                        'ai, bj, ck, dl, em -> abcdeijklm',
                        'ai, bj, ck, dl, em, fn -> abcdefijklmn',
                        'ai, bj, ck, dl, em, fn, go -> abcdefgijklmno',
                        'ai, bj, ck, dl, em, fn, go, hp -> abcdefghijklmnop'
                        )
    return N.einsum(einstein_indices[k - 1], *([frame_vecs] * k)).reshape(6**k, -1)





def probas_Pauli(k, channel, optimize='optimal'):
    """
    Yields the probability of each Pauli measurement result for 
    the scenario without ancilla.
    For a given Pauli input state and measurement basis, sums to
    one. Hence total sum is $18^k$.

    Input: k is the number of qubits,
           channel are the Kraus operators of the channel.
    Output array $(6^k, 6^k)$. First coordinate input state, second 
    coordinate measured output.
    """
    res = 0
    Pk = prod_pauli_vecs(k)
    # Looping over kraus instead of doing everything in the einsum to
    # avoid excessive memory usage if the rank is high.
    for kraus in channel:
        a = N.einsum('nj, ij, mi -> nm', Pk, kraus, Pk.conj(), optimize='optimal')
        res += (a.real**2 + a.imag**2) 
    return res
      



def M_k(k):
    """
    Yields least-square estimators components for the input (or the output)
    state.
    Output is $(6^k, 2^k, 2^k)$.
    First coordinate is the index of the state.
    Other coordinates are the corresponding matrix.
    """
    P1 = prod_pauli_vecs(1)
    # ALERT
    #
    # Here I do not understand the position of the conj(). I would have thought it is on the other P1.
    # But it is this way that yields the right result.
    M_1 = N.einsum('nd, ne -> nde', 3 * P1, P1.conj()) - N.eye(2)
    Mk = N.copy(M_1)
    for i in range(2,k+1):
        Mk = N.einsum('nde, mfg -> nmdfeg', Mk, M_1).reshape(6**i, 2**i, 2**i)
    return Mk




def Choi_LS_from_Pauli_freq(k, freq, optimize='optimal'):
    """
    Direct version, when memory is not a problem. If it is, 
    Choi_LS_Pauli_from_channel_mem will have a lighter load.
    
    Yields the least-square estimator of the Choi matrix in a Pauli setting without
    ancilla, when the result of the measurements have frequency freq.
    Input: k is the number of qubits.
           freq is the frequency of each result. Shape $(6^k, 6^k)$. Sums to the 
           number of measurement settings $18^k$.
    Output: Matrix $(4^k, 4^k)$. Trace one. Not completely positive, 
            nor trace-preserving.
        
    """
    Pk = prod_pauli_vecs(k)   
    Mk = M_k(k) 
    Choi_est = N.einsum('nm, nde, mfg -> fegd', freq, Mk, Mk, optimize=optimize) # Ordre des indices étrange 
    return Choi_est.reshape(4**k, 4**k) / 18**k




def Choi_LS_Pauli_from_channel_bigmem(k, channel, cycles=1, optimize='optimal', full_output=1):
    """
    Version where we try to keep memory in check.     
    Needs $4 * 25^k$ complex128 entries.
    Could be reduced to $16^k$ at the price of many extra calculations.

    Yields the least-square estimator of the Choi matrix in a Pauli setting without
    ancilla, when the result of the measurements have frequency freq.
    Input: k is the number of qubits.
           freq is the frequency of each result. Shape $(6^k, 6^k)$. Sums to the 
           number of measurement settings $18^k$.
    Output: Matrix $(4^k, 4^k)$. Trace one. Not completely positive, 
            nor trace-preserving.
    """
    Mk = M_k(k)
    Pk = prod_pauli_vecs(k)
    Choi_est = 0
    
    for i in range(0,6**k,4**k):
        print(f'Entering slice {i // 4**k} of {6**k // 4**k}')
        input_config_slice = slice(i, i + 4**k)
        for o in range(0,6**k,5**k):
            output_config_slice = slice(o, o + 5**k)
            image_input =  N.einsum('nj, rij -> rni', Pk[input_config_slice], channel)
            probas = 0
            for pauli_kraus in image_input:
                a = N.einsum('ni, mi -> nm', pauli_kraus, Pk[output_config_slice].conj(), optimize='optimal')
                probas = numexpr.evaluate("probas + (a.real**2 + a.imag**2)") 

            poissonized_samples = SS.poisson.rvs(probas * cycles) 
            # Strange order of indices
            Choi_est += N.einsum('nm, nde, mfg -> fegd',
                                 poissonized_samples, Mk[input_config_slice], Mk[output_config_slice], optimize=optimize)


    Choi_est = Choi_est.reshape(4**k, 4**k)
    sample_size = int(Choi_est.trace().real) 
    Choi_est /= Choi_est.trace()  
    if full_output:
        return Choi_est, sample_size  
    else:
        return Choi_est



def Choi_LS_Pauli_from_channel_mem(k, channel, cycles=1, optimize='optimal', full_output=1):
    """
    Version where we try to keep memory in check. Main impact from M_k:
    needs $24^k$ complex128 entries.
    Could be reduced to $16^k$ at the price of many extra calculations.

    Yields the least-square estimator of the Choi matrix in a Pauli setting without
    ancilla, when the result of the measurements have frequency freq.
    Input: k is the number of qubits.
           freq is the frequency of each result. Shape $(6^k, 6^k)$. Sums to the 
           number of measurement settings $18^k$.
    Output: Matrix $(4^k, 4^k)$. Trace one. Not completely positive, 
            nor trace-preserving.
    """
    Mk = M_k(k)
    Pk = prod_pauli_vecs(k)
    Choi_est = 0
    
    for i in range(0,6**k,4**k):
        input_config_slice = slice(i, i + 4**k)
        for o in range(0,6**k,4**k):
            output_config_slice = slice(o, o + 4**k)
            image_input =  N.einsum('nj, rij -> rni', Pk[input_config_slice], channel)
            probas = 0
            for pauli_kraus in image_input:
                a = N.einsum('ni, mi -> nm', pauli_kraus, Pk[output_config_slice].conj(), optimize='optimal')
                probas += (a.real**2 + a.imag**2) 

            poissonized_samples = SS.poisson.rvs(probas * cycles) 
            # Strange order of indices
            Choi_est += N.einsum('nm, nde, mfg -> fegd',
                                 poissonized_samples, Mk[input_config_slice], Mk[output_config_slice], optimize=optimize)


    Choi_est = Choi_est.reshape(4**k, 4**k)
    sample_size = int(Choi_est.trace().real) 
    Choi_est /= Choi_est.trace()  
    if full_output:
        return Choi_est, sample_size  
    else:
        return Choi_est



def sampling(probas, cycles=1, full_output=0):
    """
    Given probas and cycles, samples each outcome according to a Poisson 
    distribution with parameter $probas \times cycles$, and normalize to
    have the same sum as probas (assumed to be the right normalisation
    for further processing).

    Output the same shape as probas.

    If full_output, also returns the sample size.
    """
    samples = SS.poisson.rvs(probas * cycles)
    if full_output:
        sample_size = samples.sum()
        return samples * (probas.sum() / sample_size), sample_size
    else:
        return samples * (probas.sum() / samples.sum())
    





def oddprime(a):
     return not (a < 3 or any(a % x == 0 for x in range(2, int(a ** 0.5) + 1)))
    

def MUBS(p):
    """
    Returns the vectors of mutually unbiased bases, for $p$ an odd prime number.
    First coordinate: index of the basis in (1, p+1), as defined in Ivonovic's
    1981 article.
    Second coordinate: index of the vector in its basis
    Third coordinate: the vector
    """
    assert oddprime(p), "p must be an odd prime"
    MUBvecs = N.zeros((p+1, p, p), dtype=complex)
    MUBvecs[0] = N.eye(p) # coordinate basis
    w = N.exp(2 * N.pi * 1j / p)
    MUBvecs[p] = w**(N.outer(N.arange(p), N.arange(p))) / N.sqrt(p)
    gamma = N.arange(1,p).reshape(p-1, 1, 1)
    j = N.arange(p).reshape(1, p, 1)
    k = N.arange(p).reshape(1, 1, p)
    MUBvecs[1:p] = w**(gamma * (j + k)**2) / N.sqrt(p) # I don't know why Ivanovic uses j+k+1
    return MUBvecs




def probas_MUBS(p, channel, optimize='optimal'):
    """
    Yields the probability of each measurement setting and result  
    in the mutually unbiased bases scenario without ancilla.
    For a given MUBs input state, sums to one. Hence total sum is $p (p + 1)$.

    Input: p is the (prime) dimension of the underlying Hilbert space,
           channel are the Kraus operators of the channel.
    Output array $(p+1, p, p+1, p)$. First coordinate indexed by input basis, 
    second input vector, third measurement basis, fourth measurement (vector) result.
    """

    MUBvecs = MUBS(p)   
    probas = 0
    # Looping over kraus instead of doing everything in the einsum to
    # avoid excessive memory usage if the rank is high.
    for kraus in channel:    
        a =  N.einsum('bvj, ij, cwi -> bvcw', MUBvecs, kraus, MUBvecs.conj(), optimize=optimize)
        probas += (a.real**2 + a.imag**2) 
    probas /= (p + 1) # Same normalization as article
    return probas  





def Choi_LS_MUBS_from_freq(p, freq, optimize='optimal'):
    """
    Yields the least-square estimator of the Choi matrix in a MUBs setting without
    ancilla, when the result of the measurements have frequency freq.
    Input: k is the number of qubits.
           freq is the frequency of each result. Shape $(p+1, p, p+1, p)$. Sums to
           the number of input states $p (p+1)$.
    Output: Matrix $(4^k, 4^k)$. Trace one. Not completely positive, 
            nor trace-preserving.

    """
    MUBvecs = MUBS(p)
    M = N.einsum('bvc, bvd -> bvcd', MUBvecs, MUBvecs.conj(), optimize=optimize)
    # ALERT: Strange order of indices
    Choi1 = N.einsum('bvcw, bvde, cwfg -> fegd', freq, M, (p+1) * M / p, optimize=optimize)
    # Possible small OPTIMISATION:
    # Choi2 is not necessary if we use the same number of measurements with each input: 
    # it is then of the form Id\otimes Id and can be added to Choi4.
    #
    # We have to help einsum a little for Choi2 and Choi3
    Choi2 = N.einsum('cw, de, cwfg -> fegd', freq.sum(axis=(0,1)), - N.eye(p) / p, M, optimize=optimize)
    Choi3 = N.einsum('bv, bvde, fg -> fegd', freq.sum(axis=(2,3)), M, - N.eye(p) / p, optimize=optimize)
    Choi4 = N.eye(p**2)
    Choi_est = (Choi1 + Choi2 + Choi3).reshape(p**2, p**2) + Choi4
    return Choi_est






