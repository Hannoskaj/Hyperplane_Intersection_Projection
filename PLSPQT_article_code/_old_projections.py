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

from projections import *
from _projections_with_introspection import store_L2_distance, store_distances_all 



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

def intersection_simplified(proj_C, proj_V, p0, repetitions=100, tol=1e-7):
    """
    Makes use of <w_i|x_j> = <w_i|w_j> for all i,j.
    """
    p_i = proj_V(p0) # p0 is assumed to be in V; Added the projection in case wrong input.
    active = N.array([])
    nb_actives = 0
    XW = N.zeros((0,0))
    w_act = N.zeros([0] + list(p0.shape))
    
    for i in range(repetitions):
        proj_i = proj_C(p_i) 
        sq_norm_x_i = SL.norm(proj_i - p_i)**2
        w_i = proj_V(proj_i) - p_i
        xiwi = SL.norm(w_i)**2 # Uses the fact that <x|w> = <w|w>
        if xiwi < tol**2:
            break
        
        XW = N.column_stack([XW, N.zeros(nb_actives)])
        XW = N.row_stack([XW, N.zeros(nb_actives + 1)])
        new_xw = N.einsum('i, ki -> k', w_i.conj(), w_act).real # Notice that the scalar product are all real
                                                                # since the matrices are self-adjoint.
        XW[-1, :-1] = new_xw
        XW[:-1, -1] = new_xw.conj()
        XW[-1, -1]  = xiwi
        
        active = N.concatenate((active, [i]))
        w_act = N.concatenate([w_act, [w_i]])
    
        subset, coeffs = step(XW, sq_norm_x_i) 
        
        XW = XW[N.ix_(subset, subset)]
        active = active[subset]
        nb_actives = len(active)
        w_act = w_act[subset]
        p_i = p_i + N.einsum('k, ki -> i', coeffs, w_act) # Vérifier reshape
        
    return p_i




def jonas_projection_plus_plus(rho, maxiter=100, free_trace=True, tol=1e-7):
    
    rho_after_CP = proj_CP_threshold(rho, free_trace=False)
    rho_new = proj_TP(rho_after_CP)    
    
    dims = rho_new.shape
    def proj_C(p):
        return proj_CP_threshold(p.reshape(dims)).ravel()
    def proj_V(p):
        return proj_TP(p.reshape(dims)).ravel()
    rho_new = intersection_simplified(proj_C, proj_V, rho_new.ravel(), repetitions, tol)
    
    return rho_new.reshape(dims)
    





def intersection_simplified_with_storage(proj_C, proj_V, p0, true_Choi, group, dims, max_mem_w,
                                        maxiter=100, free_trace=True, least_ev_x_dim2_tol=1e-2, 
                                        all_dists=False, dist_L2=True, with_evs=False, t1=0, t0=0,
                                        save_intermediate=False, **kwargs):
    """
    Makes use of <w_i|x_j> = <w_i|w_j> for all i,j.
    """
    comp_time = 0
    
    loops = group.loops
    p_i = p0
    active = N.array([])
    nb_actives = 0
    XW = N.zeros((0,0))
    w_act = N.zeros([0] + list(p0.shape))
    coeffs = N.zeros((0,))
    
    for m in range(maxiter):
        
        loops.row['iteration'] = m
        loops.row['TP_proj_time'] = t1 - t0
        comp_time += t1 - t0
        if save_intermediate:
            group.rhoTP.append(p_i.reshape((1,) + dims))
        

        group.xw.append(XW.ravel())
        group.active_w.append(active)
        group.coeffs.append(coeffs)
        # max_mem_w to limit the number of w to recall (memory control)
        if nb_actives > max_mem_w:
            XW = XW[1:,1:]
            w_act = w_act[1:]
            active = active[1:]
            nb_actives -= 1        
        
        t0 = time.perf_counter()        
        proj_i, least_ev = proj_C(p_i) 
        t1 =  time.perf_counter()
        loops.row['TP_least_ev'] = least_ev
        loops.row['CP_proj_time'] = t1 - t0
        # loops.row['step_size_multiplier'] =
        comp_time += t1 - t0        

        if all_dists:
            store_distances_all(loops.row, p_i.reshape(dims) - true_Choi, prefix='TP_',
                                error_array=group.TP_evs_error, with_evs=with_evs)
            store_distances_all(loops.row, proj_i.reshape(dims) - true_Choi,  prefix='CP_',
                                with_evs=with_evs, error_array=group.CP_evs_error)
        else:
            store_L2_distance(loops.row, p_i.reshape(dims) - true_Choi, prefix='TP_')                           
            store_L2_distance(loops.row, proj_i.reshape(dims) - true_Choi, prefix='CP_')                                       
        loops.row.append()
        loops.flush()     
        
        # Breaks here because the (- least_ev) might increase on the next rho
        if  (- least_ev) < least_ev_x_dim2_tol / dims[0]:
            t1 = t0 # Do not count twice the calculation time
            break            
        
        t0 = time.perf_counter()        
        sq_norm_x_i = SL.norm(proj_i - p_i)**2
        w_i = proj_V(proj_i) - p_i
        xiwi = SL.norm(w_i)**2 # Uses the fact that <x|w> = <w|w>
        
        
        XW = N.column_stack([XW, N.zeros(nb_actives)])
        XW = N.row_stack([XW, N.zeros(nb_actives + 1)])
        new_xw = N.einsum('i, ki -> k', w_i.conj(), w_act).real # Notice that the scalar product are all real
                                                                # since the matrices are self-adjoint.
        XW[-1, :-1] = new_xw
        XW[:-1, -1] = new_xw.conj()
        XW[-1, -1]  = xiwi
        
        active = N.concatenate((active, [m]))
        w_act = N.concatenate([w_act, [w_i]])
    
        subset, coeffs = step(XW, sq_norm_x_i)         
        w_act = w_act[subset]            
        XW = XW[N.ix_(subset, subset)]
        active = active[subset]        
        nb_actives = len(subset)
        p_i = p_i + N.einsum('k, ki -> i', coeffs, w_act) # Vérifier reshape


        t1 = time.perf_counter()
               
    loops.attrs.computation_time = comp_time
    return p_i, t1 - t0, comp_time, m




def pure_HIP_with_storage(rho, loops, true_Choi, maxiter=100, free_trace=True, least_ev_x_dim2_tol=1e-2, 
                                        all_dists=False, dist_L2=True, with_evs=False, max_mem_w=1000,
                                        save_intermediate=False, **kwargs):
    
    dims = rho.shape
   
    t0 = time.perf_counter()
    rho = proj_TP(rho)
    t1 = time.perf_counter()
    
    
    def proj_C(p):
        rho_hat, least_ev = proj_CP_threshold(p.reshape(dims), free_trace=free_trace, full_output=True)
        return rho_hat.ravel(), least_ev
    def proj_V(p):
        return proj_TP(p.reshape(dims)).ravel()
    rho, dt, comp_time, m = intersection_simplified_with_storage(proj_C, proj_V, rho.ravel(), true_Choi, loops, dims,
                                        max_mem_w, maxiter, free_trace, least_ev_x_dim2_tol, 
                                        all_dists, dist_L2, with_evs, t1=t1, t0=t0, 
                                        save_intermediate=save_intermediate, **kwargs)
    
    return rho.reshape(dims),  dt, comp_time, m 
    




    
def one_step_HIP_with_storage(rho, group, true_Choi, maxiter=100, free_trace=True,
                    least_ev_x_dim2_tol=1e-2, all_dists=False, dist_L2=True, with_evs=False, 
                    save_intermediate=False, **kwargs):
    """
    least_ev_x_dim2_tol: error that will be added with the final adjustment of adding Id to get into CP.
    """
    
    loops = group.loops
    dim2 = len(rho) 
    comp_time=0
    x_sq, xiwi = -1, 1 # For the first entry in the loop. Yields the impossible -1.
    
    # rho is on CP, we first project on TP. Outside the loop because we also end on TP.
    t0 = time.perf_counter()
    rho = proj_TP(rho)
    t1 = time.perf_counter()
    
    for m in range(maxiter): 
        loops.row['iteration'] = m
        loops.row['TP_proj_time'] = t1 - t0
        comp_time += t1 - t0
        if save_intermediate:
            group.rhoTP.append(N.expand_dims(rho,0))
        

        # On CP
        t0 = time.perf_counter()
        rho_after_CP, least_ev = proj_CP_threshold(rho, free_trace, full_output=True)
        t1 = time.perf_counter()
        # Storage of statistics
        loops.row['TP_least_ev'] = least_ev
        loops.row['CP_proj_time'] = t1 - t0
        loops.row['step_size_multiplier'] = x_sq / xiwi
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
        x = rho_after_CP - rho
        rho_after_CPTP = proj_TP(rho_after_CP)
        w = rho_after_CPTP - rho
        x_sq = SL.norm(x)**2
        xiwi = SL.norm(w)**2 # Uses the fact that <x|w> = <w|w>
        rho = rho + x_sq / xiwi * w
        t1 = time.perf_counter()
        
    loops.attrs.computation_time = comp_time
    return rho, t1 - t0, comp_time, m





