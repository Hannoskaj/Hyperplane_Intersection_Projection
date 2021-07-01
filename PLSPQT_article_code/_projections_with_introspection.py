#!/usr/bin/env python
# coding: utf-8
"""
Versions of the projections to be used with generate_sampling.

Main one is :
* hyperplane_intersection_projection_switch_with_storage, implementing hyperplane 
intersection projection mixed with some alternate projections. Switch method is very 
general. It subsumes almost all the other versions of 
hyperplane_intersection_projection.

Two other are :
* the implementation of Dykstra algorithm: dykstra_projection_with_storage.
* the implementation of alternate projections algorithm: alternate_projections_with_storage.

We have also another version of the main projection, that  recalls the hyperplanes 
from before the alternate steps when reentering HIP mode:
* hyperplane_intersection_projection_recall_with_storage implementing hyperplane 
intersection projection mixed with some alternate projections. The switch is just a
fixed number of steps in each mode. 
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

from projections import *




def store_distances_all(table_row, diff_matrix, prefix='', with_evs=False, 
                        summary_row=None, summary_prefix=None, error_array=None):
    evs_error = SL.eigvalsh(diff_matrix)
    table_row[f'{prefix}dist_L2'] = N.sqrt((evs_error**2).sum())
    table_row[f'{prefix}dist_L1'] = N.abs(evs_error).sum()
    table_row[f'{prefix}dist_Linfty'] = N.maximum(evs_error.max(), - evs_error.min())
    if with_evs:
        error_array.append([evs_error])
    if summary_row is not None:
        if summary_prefix is None:
            summary_prefix = prefix
        summary_row[f'{summary_prefix}dist_L2'] = table_row[f'{prefix}dist_L2']
        summary_row[f'{summary_prefix}dist_L1'] = table_row[f'{prefix}dist_L1']
        summary_row[f'{summary_prefix}dist_Linfty'] = table_row[f'{prefix}dist_Linfty']


def store_fidelity(table_row, mat1, mat2, prefix='',  summary_row=None, 
        summary_prefix=None, error_array=None):
    sq1 = SL.sqrtm(mat1)
    fidelity = SL.sqrtm(sq1 @ mat2 @ sq1).trace() ** 2
    table_row[f'{prefix}fidelity'] = fidelity.real
    if summary_row is not None:
        if summary_prefix is None:
            summary_prefix = prefix
        summary_row[f'{summary_prefix}fidelity'] = table_row[f'{prefix}fidelity']


    
    
def store_L2_distance(table_row, diff_matrix, prefix='', **kwargs):
    table_row[f'{prefix}dist_L2'] = SL.norm(diff_matrix)





def hyperplane_intersection_projection_recall_with_storage(rho, group, true_Choi, maxiter=100, free_trace=True,
                    least_ev_x_dim2_tol=1e-2, all_dists=False, dist_L2=True, with_evs=False, 
                    save_intermediate=False, alt_steps=4, HIP_steps=20, max_mem_w=30, **kwargs):
    """ 
    Switches between alternate projections and hyperplane intersection projections, with 
    the following rules:
    * starts in alternate projections.
    * stays in alternate projections for alt_steps steps.
    * when entering hyperplane intersection mode: keeps memory of the active hyperplanes in 
    the former HIP iteration.
    * stays in hyperplane intersection projections for HIP_steps steps.

    Main input:
    * rho is the first point, assumed to be the projection on CP maps of the
    least-square estimator.
    * group is a PyTables group showing where to write the logged data in the file.
    * true_Choi is the true channel, in Choi form, for computation of statistics to 
    be logged in the file.
    * least_ev_x_dim2_tol is a stopping condition. When the least eigenvalue is 
    close enough to zero, we break. least_ev_x_dim2_tol is the absolute value of the 
    least eigenvalue multiplied by the number of eigenvalues of the channel, which 
    is an  upper bound for (half) the maximum $L^1$-loss that will be added by 
    mixing with the depolarizing channel to get into CPTP. (final_CPTP_by_mixing)
    """
    

    
    loops = group.loops
    dim2 = len(rho)
    comp_time=0
    dims = (dim2, dim2)

    active = N.array([])
    nb_actives = 0
    XW = N.zeros((0,0))
    w_act = N.zeros([0, dim2, dim2])
    target = N.array([])
    coeffs = N.array([])
    past_al = 0
    past_HIP = 0

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
        
        w =  proj_TP(rho_after_CP) - rho
        
        if past_al < alt_steps:
        # Simple alternate projection. Always needs at least one.
            past_al += 1    
            rho += w
            target -= N.einsum('ij , aij -> a', w.conj(), w_act).real

        else:
        # The intersection of hyperplanes
            past_HIP += 1
            if past_HIP >= HIP_steps:
                past_HIP = 0
                past_al = 0
            sq_norm_x = SL.norm(rho - rho_after_CP)**2            
            xiwi = SL.norm(w)**2
            
            XW = N.column_stack([XW, N.zeros(nb_actives)])
            XW = N.row_stack([XW, N.zeros(nb_actives + 1)])
            new_xw = N.einsum('ij, aij -> a', w.conj(), w_act).real # Notice that the scalar product are all real
                                                                      # since the matrices are self-adjoint.
            XW[-1, :-1] = new_xw
            XW[:-1, -1] = new_xw
            XW[-1, -1]  = xiwi
            target = N.r_[target, sq_norm_x]    

            active = N.concatenate((active, [m]))
            w_act = N.concatenate([w_act, [w]])

            subset, coeffs = step2(XW, target) 
                                  
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
            
        t1 = time.perf_counter()


    loops.attrs.computation_time = comp_time
    return rho, t1 - t0, comp_time, m
    





def step_generator(x):
    """
    Yields an iterator from a number, tuple or list, looping eternally.
    If input is already an iterator, yields it unchanged.
    """
    if N.isscalar(x):
        def _gen(x):
            while True:
                yield x
        gen = _gen(x)
    elif isinstance(x, list) or isinstance(x, tuple):
        x = list(x)
        def _gen(x):
            i = 0
            ll = len(x)
            while True:
                yield x[i % ll]
                i+=1
        gen = _gen(x)
    elif isinstance(x, collections.Iterator):
        gen = x
    return gen



def hyperplane_intersection_projection_switch_with_storage(rho, group, true_Choi, maxiter=100, free_trace=True,
                    least_ev_x_dim2_tol=1e-2, all_dists=False, dist_L2=True, with_evs=False, 
                    save_intermediate=False, HIP_to_alt_switch='first',
                    alt_to_HIP_switch='counter', min_cos = .99,
                    alt_steps=4, missing_w=1, min_part=.3, HIP_steps=10, 
                    max_mem_w=30, **kwargs):
    """ Switches between alternate projections and HIP, with the following rules:
    * starts in alternate projections.
    * stays in alternate depending on alt_to_HIP_switch:
        ** if 'counter': uses an iterator (alt_steps) of the iteration number to determine the 
        number of consecutive steps before switching. If alt_steps
        is a number, yields this number. If a list cycles on the list.
        ** if 'cos':  switching when two
        successive steps are sufficiently colinear, namely if the cosinus of
        the vectors is at least min_cos.
    * stays in HIP depending on HIP_to_alt_switch:
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
    t0 = time.perf_counter()
    rho = proj_TP(rho)
    t1 = time.perf_counter()
    

    for m in range(maxiter):
        
        print(f'Enters iteration {m}')
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
        print(f'Smallest eigenvalue is minus {-least_ev}')
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
        if (sel == 'alternate') or (m>=(maxiter-2)): # Ensures last ones are AP.
            print('Alternate projections mode')

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
            
                    group.xw.append(XW.ravel())
                    group.active_w.append(active)
                    group.coeffs.append(coeffs)
                    group.target.append(target)
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

            t1 = time.perf_counter()

        elif sel == 'HIP': # No other possibility
            print(f'HIP mode. Active hyperplanes: {1 + nb_actives}')

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
            
            group.xw.append(XW.ravel())
            group.active_w.append(active)
            group.coeffs.append(coeffs)
            group.target.append(target)


            if HIP_to_alt_switch in ['first', 'part']:
                if (subset[0] != 0) or nb_actives > max_mem_w: # max_mem_w limits memory usage
                    change = True
                elif HIP_to_alt_switch == 'part':
                    step_size = N.sqrt(N.einsum('i, ij, j', coeffs, XW, coeffs))
                    w_first_contrib = coeffs[0] * N.sqrt(XW[0,0])
                    change = (min_part * step_size >= w_first_contrib)
                else:
                    change = False
            elif  HIP_to_alt_switch in ['counter', 'missing']:
                
                # Limits memory usage
                if nb_actives > max_mem_w:
                    nb_actives -= 1
                    active = active[1:]
                    w_act = w_act[1:]
                    target = target[1:]
                    XW = XW[1:, 1:]
                    if HIP_to_alt_switch == 'missing':
                        missed += 1
                # End max_mem_w case
                
                if HIP_to_alt_switch == 'missing':
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

            t1 = time.perf_counter()

        else:
            raise ValueError('How did I get there? Typo on "HIP" or "alternate"?')
        
    loops.attrs.computation_time = comp_time
    return rho, t1 - t0, comp_time, m
    



   


def Dykstra_with_storage(rho, group, true_Choi, maxiter=100, free_trace=True, least_ev_x_dim2_tol=1e-2, 
                                        all_dists=False, dist_L2=True, with_evs=False, 
                                        save_intermediate=False, **kwargs):
    # With Dykstra, logging time cery high since I do not get for free least_ev, I need to do a separate computation.
    loops = group.loops
    dim2 = len(rho) 
    comp_time=0
    
    t0 = time.perf_counter()
    rho = proj_TP(rho)
    t1 = time.perf_counter()
        
    correction = 0
    
    for m in range(maxiter):
        
        loops.row['iteration'] = m
        loops.row['TP_proj_time'] = t1 - t0
        comp_time += t1 - t0
        if save_intermediate:
            group.rhoTP.append(N.expand_dims(rho, 0))
        
        # On CP
        t0 = time.perf_counter()
        rho_after_CP_cor = proj_CP_threshold(rho + correction, free_trace)
        t1 = time.perf_counter()
        
        least_ev = SP.linalg.lobpcg(rho, N.ones((dim2,1)), maxiter=500, tol=1e-7, largest=False)[0][0]
        
        # Storage of statistics
        loops.row['TP_least_ev'] = least_ev
        loops.row['CP_proj_time'] = t1 - t0
        loops.row['step_size_multiplier'] = 1
        comp_time += t1 - t0
        if all_dists:
            store_distances_all(loops.row, rho - true_Choi, prefix='TP_', 
                                with_evs=with_evs, error_array=group.TP_evs_error)
            store_distances_all(loops.row, rho_after_CP_cor - true_Choi, prefix='CP_', 
                                with_evs=with_evs, error_array=group.CP_evs_error)      
        else:
            store_L2_distance(loops.row, rho - true_Choi, prefix='TP_')                           
            store_L2_distance(loops.row, rho_after_CP_cor - true_Choi, prefix='CP_')                                       
        loops.row.append()
        loops.flush()   
        
        # Breaks here because the (- least_ev) might increase on the next rho
        if  (- least_ev) < least_ev_x_dim2_tol / dim2:
            t1 = t0 # Do not count twice the calculation time
            break
        
        # On TP and intersection with hyperplane
        t0 = time.perf_counter()
        correction += rho - rho_after_CP_cor
        rho = proj_TP(rho_after_CP_cor)
        t1 = time.perf_counter()

        
    loops.attrs.computation_time = comp_time
    return rho, t1 - t0, comp_time, m
    
           





def alternate_projections_with_storage(rho, group, true_Choi, maxiter=100, free_trace=True, 
                    least_ev_x_dim2_tol=1e-2, all_dists=False, dist_L2=True, with_evs=False, 
                    save_intermediate=False, **kwargs):
    loops = group.loops
    dim2 = len(rho) 
    comp_time=0
    
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
        loops.row['step_size_multiplier'] = 1
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
        
        # On TP and intersection with hyperplane
        t0 = time.perf_counter()
        rho = proj_TP(rho_after_CP)
        t1 = time.perf_counter()

        
    loops.attrs.computation_time = comp_time
    return rho, t1 - t0, comp_time, m
    
    



