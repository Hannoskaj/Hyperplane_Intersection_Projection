#!/usr/bin/env python
# coding: utf-8

"""
Utilities to generate channels defined by their Kraus operators,
and convert them to Choi state matrix if needed.

Output format is systematically an array of shape (r, d, d), 
where r is the rank of the channel, d is the dimension of the underlying
Hilbert space and each (d, d)-array is a Kraus operator.
Unless specified, 'channel' will refer to this form in the end of this 
description.

Main functions are:
* QFTKraus: Generates the channel of the quantum Fourier transform.
* KrausOp: Makes a convex combination of unitary channels. If 
no channel is provided, unitary basis is assumed. Used for generating sums 
of random unitary channels.
* add_disentanglement_noise: Takes a channel $C$ acting on qubits and returns 
a noisy version of it: after $C$, there is a chance that a projection is 
applied on the first qubit. Similar effect if used on a channel not acting on qubits.
* Choi: Generates the Choi matrix of a channel.
"""


import numpy as N
import scipy.linalg as SL
import scipy.stats as SS







def sylvester(d):
    """
    Sylvester unitary matrix.
    """
    syl = N.diagflat(N.ones(d-1), -1)
    syl[0, -1] = 1
    return syl

def clock(d):
    """
    Clock unitary matrix.
    """
    roots_unity = N.e**(N.arange(d) * 2 * N.pi * 1j / d)
    return N.diagflat(roots_unity)

def basis_unitary(d):
    """
    Yields an orthogonal basis of the set unitary matrices U(d).
    Output array is (d, d, d).
    First dimension is the index of the unitary in the basis.
    The unitary with index $di + j$ is $C^i \cdot S^j$, where
    C is the clock matrix and S is the Sylvester matrix.
    """

    clocks = clock(d)    
    clock_stack = N.eye(d, dtype=complex).reshape(1, d, d) * N.ones((d, 1, 1))
    for j in range(1, d):
        clock_stack[j,:,:] = clock_stack[j-1,:,:] @ clocks
    
    syl = sylvester(d)
    syl_stack = N.eye(d, dtype=complex).reshape(1, d, d) * N.ones((d, 1, 1))
    for j in range(1, d):
        syl_stack[j,:,:] = syl_stack[j-1,:,:] @ syl
                                                            
    
    basis = N.zeros((d**2, d, d), dtype=complex)    
    for i in range(d):
        for j in range(d):
            basis[i + j * d,:,:] = clock_stack[i,:,:] @ syl_stack[j,:,:]
    
    return basis

def sub_basis(d, indices_list):
    """
    Generates the elements of indices given in indices_list of the orthogonal 
    basis of unitary matrices given by: The unitary with 
    index $di + j$ is $C^i \cdot S^j$, where
    C is the clock matrix and S is the Sylvester matrix.

    Output array is (len(indices_list), d, d).
    """
    cl = clock(d)
    syl = sylvester(d)
    return N.array([N.linalg.matrix_power(cl, i) @ N.linalg.matrix_power(syl,j) for (i,j) in indices_list])
    

def rand_unitary(dim):
    """
    Generates a uniformly random unitary channel.
    """
    z = 1/N.sqrt(2)*(SS.norm.rvs(size=(dim,dim)) + 1j*SS.norm.rvs(size=(dim,dim)))
    q, r = SL.qr(z)
    d = r.diagonal()
    q *= d/N.abs(d)
    return q

def convex_combi_channels(d, weights, channels):
    """
    Makes a convex combination channels.

    Input:
    * d is the dimension of the underlying Hilbert space
    * weights is an array-like with the weights of each channel. They 
    must sum to one, and be non-negative.
    * channels: list of channels
    """
    weights = N.asarray(weights)
    assert N.isclose(weights.sum(), 1), "Not trace-preserving; \sum w_c[0] must equal 1."
    coeffs = N.sqrt(weights)
    Kraus =  N.concatenate([coeff * channel for (coeff, channel) \
        in zip(coeffs, channels)])
    return Kraus



def KrausOp(d, weights, indices, us=None):
    """
    Convex combination of unitary channels. 
    Write r for the rank of the operator.
    Input:
    * d is the dimension of the underlying Hilbert space
    * weights is an array-like with the weights of each channel. They 
    must sum to one, and be non-negative.
    * indices are which r unitary operators in us are chosen.
    * If the list us is None, then it is assumed to be the output basis of
    the function basis_unitary(d).
    """
    weights = N.asarray(weights)
    indices = N.asarray(indices)
    if us is None:
        us = basis_unitary(d)
    assert N.isclose(weights.sum(), 1), "Not trace-preserving; \sum w_c[0] must equal 1."
    coeffs = N.sqrt(weights) 
    Kraus =  coeffs.reshape(-1, 1, 1) * us[indices, :, :]
    return Kraus



def add_disentanglement_noise(channel, level):
    """
    Adds the following noise to a channel: with probability level, a measurement
    is applied in the natural basis to the first qubit, discarding the
    result. This corresponds to adding two Kraus operators to each Kraus 
    operator $K$, namely $P_+ K$ and $P_- K$, where $P_+$ is the projection on
    the subspace spanned by the first half of basis vectors, and $P_-$ the
    projection on the subspace spanned by the other half.

    INPUT
    channel: (r, d, d)-array of Kraus operators of the channel.
    level: Probability of applying the disentanglement. Between 0 and 1.

    OUTPUT
    In general, (2r, d, d)-array of Kraus operators. 
    First r operators are the scaled original ones.
    Last r operators are the difference between those corresponding to projecting 
    on the first half of basis vectors (as, measurement of the first qubit yielded +).
    and those corresponding to projecting on the second half 
    of basis vectors (as, measurement of the first qubit yielded -).
    Indeed the a priori rank (3r) channel is at most (2r).

    If the underlying space's dimension is odd, the second half has one more 
    dimension.
    Exception:
    * If level=0, original (r, d, d)-array of Kraus operators.
    """
    if level == 0:
        return channel
    r, d = channel.shape[:2]
    half_d = d // 2
    P_plus = N.diag(N.arange(d) < half_d) * N.sqrt(level/2)
    P_minus = N.diag(N.arange(d) >= half_d) * N.sqrt(level/2)
    proj_plus = N.einsum('ki, rij -> rkj', P_plus, channel)
    proj_minus = N.einsum('ki, rij -> rkj', P_minus, channel)
    scaled_channel = N.sqrt(1 - level/2) * channel
    return N.concatenate([scaled_channel, proj_plus - proj_minus])




def Choi(Kraus):
    """
    Takes the rank-r Kraus reprensentation of a channel
    and returns the Choi matrix of the channel.

    Input: (r, d, d)-array.
    Output $(d^2, d^2)$-array.
    """
    r, d, d = Kraus.shape 
    vecKraus = Kraus.reshape(r, d**2) 
    return N.einsum('ij, il -> jl', vecKraus / d, vecKraus.conj())




def QFTKraus(d):
    """
    Outputs the channel of the quantum Fourier transform in dimension $d$.
    """
    mult = N.outer(N.arange(d), N.arange(d))
    return N.array([N.exp(2j * N.pi * mult / d)]) / N.sqrt(d)





