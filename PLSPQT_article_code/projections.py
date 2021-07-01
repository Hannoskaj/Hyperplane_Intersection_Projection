#!/usr/bin/env python
# coding: utf-8

import numpy as N
import scipy as S
import scipy.linalg as SL
import scipy.stats as SS
import scipy.sparse as SP
import scipy.optimize as SO
import tables
import time
from pathlib import Path
import pandas
import collections
import numexpr



def proj_TP(rho):
    """
    Projects the Choi matrix rho of a channel on trace-preserving channels.
    """
    d = N.sqrt(len(rho)).astype(int)
    partial_mixed = N.eye(d) / d
    
    # N.trace on the axes corresponding to the system
    correction = N.einsum('de, fg -> dfeg',partial_mixed, (partial_mixed - N.trace(rho.reshape(4 * [d]), axis1=0, axis2=2)))
    return rho + correction.reshape(d**2, d**2)




def final_CPTP_by_mixing(rho, full_output=False):
    """
    Assumed to be in TP.
    """
    d = len(rho)
    abs_least_ev = - SL.eigvalsh(rho, subset_by_index=[0,0])[0]
    if full_output:
        return (rho + abs_least_ev * N.eye(d)) / (1 + d * abs_least_ev), - abs_least_ev
    else:
        return (rho + abs_least_ev * N.eye(d)) / (1 + d * abs_least_ev)
    




def ensure_trace(eigvals):
    """
    Assumes sum of eigvals is at least one.

    Finds the value l so that $\sum (\lambda_i - l)_+ = 1$
    and set the eigenvalues $\lambda_i$ to $(\lambda_i - l)_+$.
    """
    trace = eigvals.sum()
    while trace > 1:
        indices_positifs = eigvals.nonzero()
        l = len(indices_positifs[0]) # Number of (still) nonzero eigenvalues
        eigvals[indices_positifs] += (1 - trace) / l  
        eigvals = eigvals.clip(0)
        trace = eigvals.sum() 
    return eigvals



def new_proj_CP_threshold(rho,  free_trace=True, full_output=False, thres_least_ev=False):
    """
    If thres_least_ev=False and free_trace=False, then projects rho on CP
    trace_one operators.
    
    More generally, changes the eigenvalues without changing the eigenvectors:
    * if free_trace=True and thres_least_ev=False, then projects on CP operators,
    with no trace condition.
    * if thres_least_ev=True, free_trace is ignored. Then we threshold by minus
    the least eigenvalues before projecting on CP trace-one operator, if we
    can do that without modifying any eigenvalue by more than threshold. If we 
    cannot, we increase the largest eigenvalues by threshold, until we arrive at 
    trace one. The eigenvalue that allows passing 1 is set to the value to get a
    sum of exactly one, and all the remaining ones are set to zero.
    """
    eigvals, eigvecs = SL.eigh(rho) # Assumes hermitian; sorted from lambda_min to lambda_max
    
    least_ev = eigvals[0]
    
    if thres_least_ev:
        threshold = - least_ev # > 0
        high_indices = N.where(eigvals > threshold)
        low_indices = N.where(eigvals <= threshold)
        if (eigvals[high_indices] + threshold).sum() > 1:
            eigvals[low_indices] = 0
            eigvals[high_indices] += threshold
            eigvals = ensure_trace(eigvals)
        else:
            eigvals += threshold
            inv_cum_evs = eigvals[::-1].cumsum()[::-1]
            first_less_1 = N.where(inv_cum_evs < 1)[0][0]
            eigvals[:first_less_1 - 1] = 0
            eigvals[first_less_1 - 1] = 1 - inv_cum_evs[first_less_1]
    
    else:
        eigvals = eigvals.clip(0)
        if not free_trace:
            eigvals = ensure_trace(eigvals)
        #    
    indices_positifs = eigvals.nonzero()[0]    
    rho_hat_TLS = (eigvecs[:,indices_positifs] * eigvals[indices_positifs]) @ eigvecs[:,indices_positifs].T.conj()
    
    if full_output==2:
        return rho_hat_TLS, least_ev, len(indices_positifs)
    elif full_output:
        return rho_hat_TLS, least_ev
    else:
        return rho_hat_TLS
        



def proj_CP_threshold(rho,  free_trace=True, full_output=False, thres_least_ev=False):
    """
    If thres_least_ev=False and free_trace=False, then projects rho on CP
    trace_one operators.
    
    More generally, changes the eigenvalues without changing the eigenvectors:
    * if free_trace=True and thres_least_ev=False, then projects on CP operators,
    with no trace condition.
    * if thres_least_ev=True, free_trace is ignored. Then we bound from below all 
    eigenvalues by their original value plus the least eigenvalue (which is negative).
    Then all the lower eigenvalues take the lower bound (or zero if it is negative),
    all the higher eigenvalues are unchanged, and there is one eigenvalue in the middle
    that gets a value between its lower bound and its original value, to ensure the
    trace is one.
    """
    eigvals, eigvecs = SL.eigh(rho) # Assumes hermitian; sorted from lambda_min to lambda_max
    
    least_ev = eigvals[0]
    
    if thres_least_ev:
        threshold = - least_ev # > 0
        evlow = (eigvals - threshold).clip(0)
        toadd = eigvals - evlow
        missing = 1 - evlow.sum()
        if missing < 0: # On this rare event, revert to usual projection
            eigvals = eigvals.clip(0)
            eigvals = ensure_trace(eigvals)
        else:
            inv_cum_toadd =  toadd[::-1].cumsum()[::-1]
            last_more_missing = N.where(inv_cum_toadd >= missing)[0][-1]
            eigvals[:last_more_missing] = evlow[:last_more_missing]
            eigvals[last_more_missing] = eigvals[last_more_missing] + missing - inv_cum_toadd[last_more_missing]    
    else:
        eigvals = eigvals.clip(0)
        if not free_trace:
            eigvals = ensure_trace(eigvals)
        #    
    indices_positifs = eigvals.nonzero()[0]    
    rho_hat_TLS = (eigvecs[:,indices_positifs] * eigvals[indices_positifs]) @ eigvecs[:,indices_positifs].T.conj()
    
    if full_output==2:
        return rho_hat_TLS, least_ev, len(indices_positifs)
    elif full_output:
        return rho_hat_TLS, least_ev
    else:
        return rho_hat_TLS
        







    
def step2(XW, target):
    """
    Finds a (big) subset of hyperplanes, including the last one, such that
    the projection of the current point on the intersection of the corresponding
    half-spaces is the projection on the intersection of hyperplanes.

    Input: XW is the matrix of the scalar products between the different 
    non-normalized normal directions projected on the subspace TP, written w_i
    in the main functions.
    target is the intercept of the hyperplanes with respect to the starting point,
    on the scale given by w_i.

    Outputs which hyperplanes are kept in subset, and the coefficients on their
    respective w_i in coeffs.
    """
    nb_active = XW.shape[0]
    subset = N.array([nb_active - 1])
    coeffs = [target[-1] / XW[-1, -1]] # Always positive
    for i in range(nb_active - 2, -1, -1):
        test = (XW[i, subset].dot(coeffs) < target[i])
        # The condition to project on the intersection of the hyperplanes is that 
        # all the coefficients are non-negative. This is equivalent to belonging
        # to the normal cone to the facet.
        if test:
            subset = N.r_[i, subset]
            coeffs = SL.inv(XW[N.ix_(subset, subset)]).dot(target[subset]) 
            # Adding a new hyperplane might generate negative coefficients.
            # We remove the corresponding hyperplanes, except if it is the last 
            # hyperplane, in which case we do not add the hyperplane.
            if coeffs[-1] < 0: 
                subset = subset[1:]
                coeffs = SL.inv(XW[N.ix_(subset, subset)]).dot(target[subset]) 
            elif not N.all(coeffs >= 0):
                subset = subset[N.where(coeffs >= 0)]
                coeffs = SL.inv(XW[N.ix_(subset, subset)]).dot(target[subset])
    
    return subset, coeffs

 


def la(XWb, sq_norm_xn):
    target = N.zeros((XWb.shape[0],))
    target[-1] = sq_norm_xn
    return SL.inv(XWb).dot(target) 

def step(XW, sq_norm_xn):
    nb_active = XW.shape[0]
    subset = [nb_active - 1]
    coeffs = [sq_norm_xn / XW[-1, -1]] # Always positive
    for i in range(nb_active - 2, -1, -1):
        test = (XW[i, subset].dot(coeffs) < 0)
        if test:
            subset = [i] + subset
            coeffs = la(XW[N.ix_(subset, subset)], sq_norm_xn) # Always positive ??? Vérifier
           # assert N.all(coeffs >= 0) 
           # for now, print and correct and continue
            if not N.all(coeffs >= 0):
                print('There seems to be a negative coefficient')
                print(coeffs)
                print('The offending w is removed.')
                subset = subset[1:]
                coeffs = la(XW[N.ix_(subset, subset)], sq_norm_xn)
    return subset, coeffs







def HIP_switch(rho, HIP_to_alt_switch='first', alt_to_HIP_switch='cos', maxiter=200, depo_tol=1e-3,
        depo_rtol=1e-1, min_cos = .99, alt_steps=4, missing_w=1, min_part=.3, HIP_steps=10):

    dim2 = len(rho)
    sel = 'alternate' # Selector for the step; 'alternate' or 'HIP'.
    if alt_to_HIP_switch == 'cos':
        w_norm_ancien = N.zeros((dim2, dim2)) # Not normalized to ensure at least two steps are taken.
    elif alt_to_HIP_switch == 'counter':
        past_al = 0       # number of steps already made in 'alternate' mode.
        alt_step_gen = step_generator(alt_steps)
        current_alt_step = next(alt_step_gen)
    else:
        raise ValueError('Unknown alt_to_HIP_switch. Must be "cos" or "counter".')

    if HIP_to_alt_switch == 'counter':
        HIP_step_gen = step_generator(HIP_steps)
        past_HIP = 0
    elif HIP_to_alt_switch == 'part':
        pass
    elif HIP_to_alt_switch == 'first':
        pass
    elif HIP_to_alt_switch == 'missing':
        missed = 0    
    else:
        raise ValueError('Unknown HIP_to_alt_switch. Must be "first", "missing", "part" or "counter".')



    dims = (dim2, dim2)

    active = N.array([])
    nb_actives = 0
    XW = N.zeros((0,0))
    w_act = N.zeros([0, dim2, dim2])
    target = N.array([])
    coeffs = N.array([])
    
    # rho is on CP, we first project on TP. Outside the loop because we also end on TP.
    rho = proj_TP(rho)
    
    for m in range(maxiter):

        # On CP
        rho_after_CP, least_ev = proj_CP_threshold(rho, free_trace, full_output=True)
        
        # Breaks here because the (- least_ev) might increase on the next rho
        if  (- least_ev) < least_ev_x_dim2_tol / dim2:
            break
            
        if sel == 'alternate':
                        
            # On TP and intersection with hyperplane
            if alt_to_HIP_switch == 'cos':
                w_new = proj_TP(rho_after_CP) - rho
                norm_w = SL.norm(w_new)
                change = (N.vdot(w_new / norm_w, w_norm_ancien).real > min_cos)
                w_norm_ancien = w_new / norm_w

                # If change with alt_steps, the current projection is transformed into
                # the first HIP step.
                if change:
                    active = N.array([m])
                    nb_actives = 1
                    XW = N.array([[norm_w**2]])
                    w_act = N.array([w_new])
                    coeffs = N.array([SL.norm(rho - rho_after_CP)**2 / norm_w**2])
                    target = N.array([0.])
                    rho += coeffs[0] * w_new
            
                else:
                    rho += w_new
                    
            elif alt_to_HIP_switch == 'counter':
                rho = proj_TP(rho_after_CP)
                past_al += 1
                change = (past_al >= current_alt_step)

                if change:
                    active = N.array([])
                    nb_actives = 0
                    XW = N.zeros((0,0))
                    w_act = N.zeros([0, dim2, dim2])
                    target = N.array([])
                    coeffs = N.array([])

            if change:
                if HIP_to_alt_switch == 'missing':
                    missed = 0
                elif HIP_to_alt_switch == 'counter':
                    past_HIP = 0
                    current_HIP_step = next(HIP_step_gen)
                sel = 'HIP'


        elif sel == 'HIP': # No other possibility
            
            sq_norm_x_i = SL.norm(rho - rho_after_CP)**2
            w_i =  proj_TP(rho_after_CP) - rho
            xiwi = SL.norm(w_i)**2
            
            XW = N.column_stack([XW, N.zeros(nb_actives)])
            XW = N.row_stack([XW, N.zeros(nb_actives + 1)])
            new_xw = N.einsum('ij, kij -> k', w_i.conj(), w_act).real # Notice that the scalar product are all real
                                                                      # since the matrices are self-adjoint.
            XW[-1, :-1] = new_xw
            XW[:-1, -1] = new_xw
            XW[-1, -1]  = xiwi
            target = N.r_[target, sq_norm_x_i]    
            
        
            active = N.concatenate((active, [m]))
            w_act = N.concatenate([w_act, [w_i]])

            subset, coeffs = step2(XW, target) 
            
            if HIP_to_alt_switch == 'missing':
                missed += len(active) - len(subset) # Don't move this after the update to active !!!
                                 
            XW = XW[N.ix_(subset, subset)]
            active = active[subset]
            nb_actives = len(active)
            w_act = w_act[subset]
            target = N.zeros((nb_actives,))
            rho += N.einsum('k, kij -> ij', coeffs, w_act)

            if HIP_to_alt_switch in ['first', 'part']:
                if (subset[0] != 0):
                    change = True
                elif HIP_to_alt_switch == 'part':
                    step_size = N.sqrt(N.einsum('i, ij, j', coeffs, XW, coeffs))
                    w_first_contrib = coeffs[0] * N.sqrt(XW[0,0])
                    change = (min_part * step_size >= w_first_contrib)
                else:
                    change = False
            elif HIP_to_alt_switch == 'missing':
                change = (missed >= missing_w)
            elif HIP_to_alt_switch == 'counter':
                past_HIP += 1
                change = (past_HIP >= current_HIP_step)    

            if change:
                if alt_to_HIP_switch == 'cos':
                    w_norm_ancien = N.zeros((dim2, dim2)) # Ensures two alternate steps. Also possible to
                                                          # use w_norm_ancien = w_i / N.sqrt(xiwi)
                elif alt_to_HIP_switch == 'counter':
                    past_al = 0
                    current_alt_step = next(alt_step_gen)
                sel = 'alternate'

    return rho
    



def increasing_steps(step, start=None, maxi=N.inf):
    """
    Yields a generator, increasing each output by the same. 
    If start is given, starts at start instead of step.
    If maxi is given, always yields maxi when the sum of steps exceeds maxi.
    """
    if start is None:
        res = step
    else:
        res = start
    while res < maxi:
        yield res
        res += step
    while True:
        yield maxi


