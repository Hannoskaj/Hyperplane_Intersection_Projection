#!/usr/bin/env python
# coding: utf-8
"""
Older projections or potentially interesting implementatons.
"""


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




def prod_pauli_vecs_for_subsets(k, U2=None):
    """
    Outputs all the k-tensor products of Pauli vectors, as an array 
    where the dimensions are: bases * base elements * vector coordinates.
    Works till k=8 included. Needs $12^k$ complex entries.
    U2 allows to add a rotation to the Pauli vectors, so as to avoid very special cases.
    """
    s2 = N.sqrt(.5)
    frame_vecs = N.array(([[1,0], [0,1]], [[s2, s2], [s2, -s2]], [[s2, s2 * 1j], [s2, -s2 *1j]]))
    if U2 is not None:
        frame_vecs = N.dot(frame_vecs, U2)
    einstein_indices = ('aiq -> aiq',
                        'aiq, bjr -> abijqr',
                        'aiq, bjr, cks -> abcijkqrs',
                        'aiq, bjr, cks, dlt -> abcdijklqrst',
                        'aiq, bjr, cks, dlt, emu -> abcdeijklmqrstu',
                        'aiq, bjr, cks, dlt, emu, fnv -> abcdefijklmnqrstuv',
                        'aiq, bjr, cks, dlt, emu, fnv, gow -> abcdefhijklmnoqrtstuvw'
                        'aiq, bjr, cks, dlt, emu, fnv, gow, hpx -> abcdefghijklmnopqrstuvwx'
                        )
    return N.einsum(einstein_indices[k - 1], *([frame_vecs] * k)).reshape(3**k, 2**k, -1)



def probas_Pauli_ancien(k, Kraus, optimize='optimal'):
    Pk = prod_pauli_vecs(k)
    images = N.einsum('nj, rij -> nri', Pk, Kraus)
    probas = N.einsum('nrd, nre, md, me -> nm', images, images.conj(), Pk.conj(), Pk, optimize=optimize).real
    return probas.clip(0) # Avoids the -1e-17 that can happen with floats


def probas_Pauli_direct(k, Kraus, optimize='optimal'):
    Pk = prod_pauli_vecs(k)
    a = N.einsum('nj, rij, mi -> rnm', Pk, Kraus, Pk.conj(), optimize=optimize)
    probas = (a.real**2 + a.imag**2).sum(0) 
    return probas  





def probas_MUBS_ancien(p, Kraus, optimize='optimal'):
    MUBvecs = MUBS(p)
    images = N.einsum('bvc, rco -> bvro', MUBvecs, Kraus)
    return N.einsum('bnrd, bnre, cmd, cme -> bncm', images, images.conj(), MUBvecs, MUBvecs.conj(), optimize=optimize).real




def probas_MUBS(p, Kraus, optimize='optimal'):
    MUBvecs = MUBS(p) 
    a =  N.einsum('bvj, rij, cwi -> rbvcw', MUBvecs, Kraus, MUBvecs.conj(), optimize=optimize)
    probas = (a.real**2 + a.imag**2).sum(0) / (p + 1)
    return probas  



def Choi_LS_Pauli_mem(k, channel, cycles=1, optimize='optimal', full_output=1):
    """
    WITHOUT CONTROL ON RANK
    
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
    images = N.einsum('nj, rij -> nri', Pk, channel)
    Choi_est = 0
    
    for i in range(0,6**k,4**k):
        input_config_slice = slice(i, i + 4**k)
        for o in range(0,6**k,4**k):
            output_config_slice = slice(o, o + 4**k)
            probas = SL.norm(N.einsum('nre, me -> rnm', 
                                      images[input_config_slice], Pk[output_config_slice].conj(), optimize=optimize),
                             axis=0)**2
            poissonized_samples = SS.poisson.rvs(probas * cycles) 
            # Strange order of indices
            Choi_est += N.einsum('nm, nde, mfg -> fegd',
                                 poissonized_samples, Mk[input_config_slice], Mk[output_config_slice], optimize=optimize)


    Choi_est = Choi_est.reshape(4**k, 4**k)
    sample_size = Choi_est.trace().real * 18**k
    Choi_est /= Choi_est.trace()  
    return Choi_est, sample_size





def hyperplane_intersection_projection_switch_with_storage(rho, group, true_Choi, maxiter=100, free_trace=True,
                    least_ev_x_dim2_tol=1e-2, all_dists=False, dist_L2=True, with_evs=False, 
                    save_intermediate=False, HIP_to_alt_switch='counter',
                    alt_to_HIP_switch='counter', min_cos = .99,
                    alt_steps=6, missing_w=1, min_part=.3, HIP_steps=40, 
                    keep_mem_w=True, max_mem_w=30, **kwargs):
    """ 
    Switches between alternate projections and hyperplane intersection projections, with 
    the following rules:
    * starts in alternate projections.
    * stays in alternate depending on alt_to_HIP_switch:
        ** if 'counter': uses an iterator (alt_steps) of the iteration number to determine the 
        number of consecutive steps before switching. If alt_steps
        is a number, yields this number. If a list cycles on the list.
        ** if 'cos':  switching when two
        successive steps are sufficiently colinear, namely if the cosinus of
        the vectors is at least min_cos.
    * when entering hyperplane intersection mode: keeps memory of the active hyperplanes in 
    the former HIP iteration if keep_mem_w set to True, otherwise starts with an empty set of
    hyperplanes in memory.
    * stays in hyperplane intersection depending on HIP_to_alt_switch:
        ** if 'first': stops HIP when the first active hyperplane
        of the sequence gets discarded. (ex: enter at iteration 7, then leaves when 
        the hyperplane of iteration 7 is not in w_act anymore).
        ** if 'missing', stops when a total of missing_w (default 1) hyperplanes are 
        deemed unnecessary. (ie w_act has lost missing_w member).
        ** if 'part': ends the loop if the length coeff_first * w_first is less than min_part 
        times the step size, ie the length of \sum coeffs_i w_i. This includes the case when
        the first hyperplane is deemed unnecessary, like in 'first'.
        ** if 'counter': uses an iterator (HIP_steps) of the iteration number to determine the 
        number of consecutive steps before switching. Iterator in input iter_choice. If 
        HIP_steps is a number, yields this number. If a list cycles on the list.
    """
    


    loops = group.loops
    dim2 = len(rho) 
    comp_time=0
    # x_sq, xiwi = -1, 1 # For the first entry in the loop. Yields the impossible -1.
    sel = 'alternate' # Selector for the step; 'alternate' or 'HIP'.
    if alt_to_HIP_switch == 'cos':
        w_norm_ancien = N.zeros((dim2, dim2)) # Not normalized to ensure at least two steps are taken.
    elif alt_to_HIP_switch == 'counter':
        past_al = 0       # number of steps already made in 'alternate' mode.
        alt_step_gen = step_generator(alt_steps) # Allows more general conditions
                                                 # than fixed number of steps.
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
    t0 = time.perf_counter()
    rho = proj_TP(rho)
    t1 = time.perf_counter()
    

    for m in range(maxiter):

        loops.row['iteration'] = m
        loops.row['TP_proj_time'] = t1 - t0
        comp_time += t1 - t0
            
        # On CP
        t0 = time.perf_counter()
        rho_after_CP, least_ev = proj_CP_threshold(rho, free_trace, full_output=True)
        t1 = time.perf_counter()
        if save_intermediate:
            group.rhoTP.append(N.expand_dims(rho, 0))
        
        # Storage of statistics
        loops.row['TP_least_ev'] = least_ev
        loops.row['CP_proj_time'] = t1 - t0
        # loops.row['step_size_multiplier'] = ssm
        comp_time += t1 - t0
        if all_dists:
            store_distances_all(loops.row, rho - true_Choi, prefix='TP_', 
                                with_evs=with_evs, error_array=group.TP_evs_error)
            store_distances_all(loops.row, rho_after_CP - true_Choi, prefix='CP_', 
                                with_evs=with_evs, error_array=group.CP_evs_error)      
        else:
            store_L2_distance(loops.row, rho - true_Choi, prefix='TP_')                           
            store_L2_distance(loops.row, rho_after_CP - true_Choi, prefix='CP_')                                       
        loops.row.append()
        loops.flush()   
        
        # Breaks here because the (- least_ev) might increase on the next rho
        if  (- least_ev) < least_ev_x_dim2_tol / dim2:
            t1 = t0 # Do not count twice the calculation time
            break
            
        t0 = time.perf_counter()       
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
                    rho += coeffs[0] * w_new # TO BE COMPLETED
            
                    group.xw.append(XW.ravel())
                    group.active_w.append(active)
                    group.coeffs.append(coeffs)
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
                    
            if change:
                if HIP_to_alt_switch == 'missing':
                    missed = 0
                elif HIP_to_alt_switch == 'counter':
                    past_HIP = 0
                    current_HIP_step = next(HIP_step_gen)
                sel = 'HIP'

            t1 = time.perf_counter()

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

            subset, coeffs = step2(XW, target, max_mem_w) 
            
            if HIP_to_alt_switch == 'missing':
                missed += len(active) - len(subset) # Don't move this after the update to active !!!
                                 
            XW = XW[N.ix_(subset, subset)]
            active = active[subset]
            nb_actives = len(active)
            w_act = w_act[subset]
            target = N.zeros((nb_actives,))
            rho += N.einsum('k, kij -> ij', coeffs, w_act)
            
            group.xw.append(XW.ravel())
            group.active_w.append(active)
            group.coeffs.append(coeffs)
            group.target.append(target)


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
                    w_norm_ancien = w_i / SL.norm(w_i)
                
                elif alt_to_HIP_switch == 'counter':
                    past_al = 0
                    current_alt_step = next(alt_step_gen)
                sel = 'alternate'

            t1 = time.perf_counter()

        else:
            raise ValueError('How did I get there? Typo on "HIP" or "alternate"?')
        
    loops.attrs.computation_time = comp_time
    return rho, t1 - t0, comp_time, m
    





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

