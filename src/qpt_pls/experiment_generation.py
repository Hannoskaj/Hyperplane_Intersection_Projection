#!/usr/bin/env python
# coding: utf-8
"""
The function generate_simulations to generate experiments.

I have added the function retrieve_main to retrieve the most relevant data in the
created files, in a more easy format.


Saves results via PyTables.
Main structure:
    * /summary
    * /exp_repet{number}
    ** /exp_repet{number}/initialisation
    ** /exp_repet{number}/{projection}
    ** Also the eigenvalues of the LS and CP estimators.
    *** /exp_repet{number}/{projection}/loops
    *** Also much inner information at this level, like the set of active hyperplanes.

summary contains the distances between target and LS estimator, first CP estimator,
    final pre-mixing estimator, final estimator; as well as computation time,
    without the logging and data generation time, all for each repetition and
    projection.
summary.attrs.RandomState contains the original random state

initialisation contains the information relative to the LS and CP estimators.
initialisation.attrs.RandomState contains the random state at the start of this 
    experiment repetition.

loops contains information on each loop of the projection, in particular 
    the least eigenvalue of the TP estimator, least_ev, used for convergence.
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

from qpt_pls._projections_with_introspection import (
    hyperplane_intersection_projection_recall_with_storage,
    hyperplane_intersection_projection_switch_with_storage,
    step_generator,
    Dykstra_with_storage,
    alternate_projections_with_storage,
    store_distances_all,
    store_L2_distance,
    store_fidelity,
)

from qpt_pls.data_generation import (
    sampling,
    probas_Pauli,
    probas_MUBS,
    Choi_LS_Pauli_from_channel_mem,
    Choi_LS_MUBS_from_freq,
    Choi_LS_from_Pauli_freq,
    Choi_LS_Pauli_from_channel_bigmem,
)

from qpt_pls._old_projections import one_step_HIP_with_storage, pure_HIP_with_storage

from qpt_pls.channels import Choi

from qpt_pls.projections import *

# from old_ideas import probas_Pauli_ancien
# from data_generation import prod_pauli_vecs
# def probas_Pauli_ancien(k, Kraus, optimize='optimal'):
#    Pk = prod_pauli_vecs(k)
#    images = N.einsum('nj, rij -> nri', Pk, Kraus)
#    probas = N.einsum('nrd, nre, md, me -> nm', images, images.conj(), Pk.conj(), Pk, optimize=optimize).real
#    return probas.clip(0) # Avoids the -1e-17 that can happen with floats


def summary(projection):
    Summary = {
        "time_exp": tables.Float64Col(dflt=-1),
        "sample_size": tables.Int64Col(dflt=-1),
        "LS_dist_L2": tables.Float64Col(dflt=-1),
        "LS_dist_L1": tables.Float64Col(dflt=-1),
        "LS_dist_Linfty": tables.Float64Col(dflt=-1),
        "CP_dist_L2": tables.Float64Col(dflt=-1),
        "CP_dist_L1": tables.Float64Col(dflt=-1),
        "CP_dist_Linfty": tables.Float64Col(dflt=-1),
    }
    for proj in projection:
        Summary[f"{proj}_number_of_iterations"] = tables.Int64Col(dflt=-1)
        Summary[f"{proj}_TPfinal_dist_L2"] = tables.Float64Col(dflt=-1)
        Summary[f"{proj}_TPfinal_dist_L1"] = tables.Float64Col(dflt=-1)
        Summary[f"{proj}_TPfinal_dist_Linfty"] = tables.Float64Col(dflt=-1)
        Summary[f"{proj}_final_dist_L2"] = tables.Float64Col(dflt=-1)
        Summary[f"{proj}_final_dist_L1"] = tables.Float64Col(dflt=-1)
        Summary[f"{proj}_final_dist_Linfty"] = tables.Float64Col(dflt=-1)
        Summary[f"{proj}_max_active_w"] = tables.Float64Col(dflt=-1)

    return Summary


# LS_least_ev
# LS_sum_of_square_neg_evs


def setup_info(dim):
    class Setup_info(tables.IsDescription):
        """
  Has the attribute RandomState storing the state of the rng at the start of the experiment.
  """

        data_generation_and_LS_time = tables.Float64Col(dflt=-1)
        CP_proj_time = tables.Float64Col(dflt=-1)
        sample_size = tables.Int64Col(dflt=-1)
        LS_dist_L2 = tables.Float64Col(dflt=-1)
        LS_dist_L1 = tables.Float64Col(dflt=-1)
        LS_dist_Linfty = tables.Float64Col(dflt=-1)
        LS_fidelity = tables.Float64Col(dflt=-1)
        LS_least_ev = tables.Float64Col(dflt=-1)
        CP_dist_L2 = tables.Float64Col(dflt=-1)
        CP_dist_L1 = tables.Float64Col(dflt=-1)
        CP_dist_Linfty = tables.Float64Col(dflt=-1)
        CP_fidelity = tables.Float64Col(dflt=-1)
        # Needs to add arrays. Otherwise pbs when dim big.
        # LS_evs_error = tables.Float64Col(dflt=10, shape=(dim**2,))
        # CP_evs_error = tables.Float64Col(dflt=10, shape=(dim**2,))
        target_abs_least_ev = tables.Float64Col(dflt=-1)

    return Setup_info


# LS_sum_of_square_neg_evs


def after_tp_proj(dim, all_dists=False, dist_L2=True, with_evs=False):

    After_TP_proj = {
        # Defaults chosen as impossible values.
        "iteration": tables.Int64Col(dflt=-1),
        "CP_proj_time": tables.Float64Col(dflt=-1),
        "TP_proj_time": tables.Float64Col(dflt=-1),
        "TP_least_ev": tables.Float64Col(dflt=-1),
        "step_size_multiplier": tables.Float64Col(dflt=-1),
    }

    if all_dists:
        After_TP_proj["TP_dist_L2"] = tables.Float64Col(dflt=-1)
        After_TP_proj["TP_dist_L1"] = tables.Float64Col(dflt=-1)
        After_TP_proj["TP_dist_Linfty"] = tables.Float64Col(dflt=-1)
        After_TP_proj["CP_dist_L2"] = tables.Float64Col(dflt=-1)
        After_TP_proj["CP_dist_L1"] = tables.Float64Col(dflt=-1)
        After_TP_proj["CP_dist_Linfty"] = tables.Float64Col(dflt=-1)
        # Needs to add arrays. Otherwise pbs when dim big.
        # if with_evs:
        #    After_TP_proj['TP_evs_error'] = tables.Float64Col(dflt=10, shape=(dim**2,))
        #    After_TP_proj['CP_evs_error'] = tables.Float64Col(dflt=10, shape=(dim**2,))
    elif dist_L2:
        After_TP_proj["TP_dist_L2"] = tables.Float64Col(dflt=-1)
        After_TP_proj["CP_dist_L2"] = tables.Float64Col(dflt=-1)

    return After_TP_proj


# TP_sum_of_square_neg_evs


def final_state(dim):
    class Final_state(tables.IsDescription):
        number_of_iterations = tables.Int64Col(dflt=-1)
        final_proj_time = tables.Float64Col(dflt=-1)
        TP_proj_time = tables.Float64Col(dflt=-1)
        total_time = tables.Float64Col(dflt=-1, pos=1)
        TPfinal_dist_L2 = tables.Float64Col(dflt=-1)
        TPfinal_dist_L1 = tables.Float64Col(dflt=-1)
        TPfinal_dist_Linfty = tables.Float64Col(dflt=-1)
        TPfinal_fidelity = tables.Float64Col(dflt=-1)
        TPfinal_least_ev = tables.Float64Col(dflt=-1)
        final_dist_L2 = tables.Float64Col(dflt=-1)
        final_dist_L1 = tables.Float64Col(dflt=-1)
        final_dist_Linfty = tables.Float64Col(dflt=-1)
        final_fidelity = tables.Float64Col(dflt=-1)
        # Needs to add arrays. Otherwise pbs when dim big.
        # TPfinal_evs_error = tables.Float64Col(dflt=10, shape=(dim**2,))
        # final_evs_error = tables.Float64Col(dflt=10, shape=(dim**2,))

    return Final_state


# TP_sum_of_square_neg_evs
# TP_least_ev

# ADDED steps to avoid trying to pickle a generator.
def _generate_simulations(
    fileh,
    dim,
    channel,
    scenario="Pauli",
    cycles=1,
    projections=["HIPswitch"],
    repetitions=1,
    options_proj={"alt_steps": 10, "HIP_steps": 50, "maxiter": 100, "max_mem_w": 30},
    steps={"genfun_alt": None, "genfun_HIP": None},
    depo_tol=1e-3,
    depo_rtol=1e-1,
    first_CP_threshold_least_ev=True,
    all_dists=False,
    dist_L2=True,
    with_evs=False,
    keep_key_channels=False,
    keep_main_eigenvectors=0,
    save_intermediate=False,
    bigmem=False,
    random_state=None,
):

    t_init = time.perf_counter()
    # Setting up RNG
    if random_state is not None:
        try:
            N.random.set_state(random_state)
        except:
            rs = S._lib._util.check_random_state(random_state)
            N.random.set_state(rs.get_state())

    # Controlling inputs and defining global variables
    Summary = summary(projections)
    Setup_info = setup_info(dim)
    After_TP_proj = after_tp_proj(
        dim, all_dists=all_dists, dist_L2=dist_L2, with_evs=with_evs
    )
    Final_state = final_state(dim)
    compute_time = 0

    true_Choi = Choi(channel)

    if keep_main_eigenvectors != 0:
        main_eigvals, main_eigvecs = SL.eigh(
            true_Choi, subset_by_index=[dim ** 2 - keep_main_eigenvectors, dim ** 2 - 1]
        )
        fileh.create_array(
            "/",
            "main_eigvals_true",
            main_eigvals,
            "Main eigenvalues of the true channel.",
        )
        fileh.create_array(
            "/",
            "main_eigvecs_true",
            main_eigvecs,
            "Main eigenvectors of the true channel.",
        )

    if scenario == "Pauli":
        k = int(N.log2(dim))
        mean_cycle_size = 18 ** k
        if bigmem:
            if k < 7:
                probas = probas_Pauli(
                    k, channel
                )  # For reproductibility with old versions, use
                # probas_Pauli_ancien instead. The 1e-17 counts.
                def Choi_LS():
                    freq, sample_size = sampling(probas, cycles, full_output=1)
                    return Choi_LS_from_Pauli_freq(k, freq), sample_size

            else:

                def Choi_LS():
                    return Choi_LS_Pauli_from_channel_bigmem(
                        k, channel, cycles, full_output=1
                    )

        else:
            if k < 6:
                probas = probas_Pauli(
                    k, channel
                )  # For reproductibility with old versions, use
                # probas_Pauli_ancien instead. The 1e-17 counts.
                def Choi_LS():
                    freq, sample_size = sampling(probas, cycles, full_output=1)
                    return Choi_LS_from_Pauli_freq(k, freq), sample_size

            else:

                def Choi_LS():
                    return Choi_LS_Pauli_from_channel_mem(
                        k, channel, cycles, full_output=1
                    )

    elif scenario == "MUBs":
        mean_cycle_size = (dim + 1) * dim
        probas = probas_MUBS(dim, channel)

        def Choi_LS():
            freq, sample_size = sampling(probas, cycles, full_output=1)
            return Choi_LS_MUBS_from_freq(dim, freq), sample_size

    else:
        raise ValueError('unknown scenario. Should be "Pauli" or "MUBs"')
    #
    #
    #

    # Setting up the storage
    root = fileh.root
    table_summary = fileh.create_table(
        root, "summary", Summary, "Summary of all repetitions"
    )
    # Stores target channel
    fileh.create_array(
        root, "kraus_true_channel", channel, "Kraus operators of the true channel"
    )
    for repet in range(repetitions):
        exp_repet = fileh.create_group(
            root, f"exp_repet{repet}", f"Repetition number {repet} of the experiment"
        )
        fileh.create_table(exp_repet, "initialisation", Setup_info, "Setup information")
        fileh.create_earray(
            exp_repet,
            "LS_evs_error",
            tables.Float64Atom(),
            (0, dim ** 2),
            "Eigenvalues of the difference between the LS estimator and the true channel",
        )
        fileh.create_earray(
            exp_repet,
            "CP_evs_error",
            tables.Float64Atom(),
            (0, dim ** 2),
            "Eigenvalues of the difference between the CP estimator and the true channel",
        )
        for proj in projections:
            projection = fileh.create_group(
                exp_repet, "projection_" + proj, f"Results for projection {proj}"
            )
            fileh.create_table(
                projection,
                "loops",
                After_TP_proj,
                "Distances and information after the TP projection",
            )
            fileh.create_table(
                projection,
                "end",
                Final_state,
                "Distances and information at the last stage",
            )
            fileh.create_earray(
                projection,
                "TPfinal_evs_error",
                tables.Float64Atom(),
                (0, dim ** 2),
                "Eigenvalues of the difference between the last TP estimator and the true channel",
            )
            fileh.create_earray(
                projection,
                "final_evs_error",
                tables.Float64Atom(),
                (0, dim ** 2),
                "Eigenvalues of the difference between the final estimator and the true channel",
            )
            if all_dists:
                fileh.create_earray(
                    projection,
                    "TP_evs_error",
                    tables.Float64Atom(),
                    (0, dim ** 2),
                    "Eigenvalues of the difference between the TP estimator and the true channel",
                )
                fileh.create_earray(
                    projection,
                    "CP_evs_error",
                    tables.Float64Atom(),
                    (0, dim ** 2),
                    "Eigenvalues of the difference between the CP estimator and the true channel",
                )
            if save_intermediate:
                fileh.create_earray(
                    projection,
                    "rhoTP",
                    tables.ComplexAtom(itemsize=16),
                    (0, dim ** 2, dim ** 2),
                    "Intermediate TP estimators",
                )
            if proj in ["pureHIP", "HIPswitch", "HIPrec"]:
                fileh.create_vlarray(
                    projection,
                    "active_w",
                    tables.Int32Atom(),
                    "Iteration numbers of the hyperplanes used in approximating CP",
                )
                fileh.create_vlarray(
                    projection,
                    "xw",
                    tables.Float64Atom(),
                    "Raveled scalar products between steps of the hyperplanes used in approximating CP",
                )
                fileh.create_vlarray(
                    projection,
                    "coeffs",
                    tables.Float64Atom(),
                    "Coefficients for the non-normalized directions for the new TP",
                )
            if proj in ["HIPswitch", "HIPrec"]:
                fileh.create_vlarray(
                    projection,
                    "target",
                    tables.Float64Atom(),
                    "Target for the intersection of hyperplanes",
                )

    #
    #
    #

    # Storing information about experiment
    table_summary.attrs.RandomState = N.random.get_state()
    table_summary.attrs.date = time.localtime()
    table_summary.attrs.scenario = scenario
    table_summary.attrs.cycles = cycles
    table_summary.attrs.dim = dim
    table_summary.attrs.repetitions = repetitions
    table_summary.attrs.projections = projections
    table_summary.attrs.options_proj = options_proj
    table_summary.attrs.expected_sample_size = cycles * mean_cycle_size
    table_summary.attrs.first_CP_threshold_least_ev = first_CP_threshold_least_ev
    table_summary.attrs.keep_key_channels = keep_key_channels
    table_summary.attrs.all_dists = all_dists
    table_summary.attrs.with_evs = with_evs
    table_summary.attrs.dist_L2 = dist_L2
    #
    #
    #

    # Experiment repetitions
    for repet in range(repetitions):
        time_exp = 0
        exp_repet = fileh.get_node(f"/exp_repet{repet}")
        exp_repet.initialisation.attrs.RandomState = N.random.get_state()
        exp_repet.initialisation.attrs.date = time.localtime()

        # Data generation and computation of least-square estimator
        t0 = time.perf_counter()
        rho, sample_size = Choi_LS()
        t1 = time.perf_counter()

        # Storing corresponding statistics
        init = exp_repet.initialisation.row
        init["data_generation_and_LS_time"] = t1 - t0
        time_exp += t1 - t0
        init["sample_size"] = sample_size
        table_summary.row["sample_size"] = sample_size

        store_distances_all(
            init,
            rho - true_Choi,
            prefix="LS_",
            with_evs=True,
            error_array=exp_repet.LS_evs_error,
            summary_row=table_summary.row,
        )
        store_fidelity(init, rho, true_Choi, prefix="LS_")
        if keep_key_channels:
            fileh.create_array(
                exp_repet,
                "LS_estimator",
                rho,
                "Least-square estimator of the Choi matrix of the channel",
            )
        if keep_main_eigenvectors != 0:
            main_eigvals, main_eigvecs = SL.eigh(
                rho, subset_by_index=[dim ** 2 - keep_main_eigenvectors, dim ** 2 - 1]
            )
            fileh.create_array(
                exp_repet,
                "main_eigvals_LS",
                main_eigvals,
                "Main eigenvalues of the LS estimator.",
            )
            fileh.create_array(
                exp_repet,
                "main_eigvecs_LS",
                main_eigvecs,
                "Main eigenvectors of the LS estimator.",
            )

        # CP estimator: first projection on CP matrices
        t0 = time.perf_counter()
        rhoCP, LS_least_ev = proj_CP_threshold(
            rho, full_output=True, thres_least_ev=first_CP_threshold_least_ev
        )
        t1 = time.perf_counter()

        # Setting up the tolerance for the least eigenvalue
        ls_rel = -LS_least_ev * depo_rtol
        least_ev_x_dim2_tol = N.maximum(ls_rel, depo_tol)
        init["target_abs_least_ev"] = least_ev_x_dim2_tol / dim ** 2

        # Storing corresponding statistics
        init["LS_least_ev"] = LS_least_ev
        init["CP_proj_time"] = t1 - t0
        time_exp += t1 - t0

        store_distances_all(
            init,
            rhoCP - true_Choi,
            prefix="CP_",
            with_evs=True,
            error_array=exp_repet.CP_evs_error,
            summary_row=table_summary.row,
        )
        store_fidelity(init, rhoCP, true_Choi, prefix="CP_")
        if keep_key_channels:
            fileh.create_array(
                exp_repet,
                "CP_estimator",
                rhoCP,
                "First CP estimator of the Choi matrix of the channel",
            )
        if keep_main_eigenvectors != 0:
            main_eigvals, main_eigvecs = SL.eigh(
                rhoCP, subset_by_index=[dim ** 2 - keep_main_eigenvectors, dim ** 2 - 1]
            )
            fileh.create_array(
                exp_repet,
                "main_eigvals_CP",
                main_eigvals,
                "Main eigenvalues of the CP estimator.",
            )
            fileh.create_array(
                exp_repet,
                "main_eigvecs_CP",
                main_eigvecs,
                "Main eigenvectors of the CP estimator.",
            )

        # End initialisation
        init.append()
        exp_repet.initialisation.flush()

        for proj in projections:
            group = fileh.get_node(f"/exp_repet{repet}/projection_{proj}")
            # REINITIALISING A POTENTIAL GENERATOR
            #
            if "genfun_alt" in steps and steps["genfun_alt"] is not None:
                options_proj["alt_steps"] = steps["genfun_alt"](
                    *options_proj["genarg_alt"]
                )
            if "genfun_HIP" in steps and steps["genfun_HIP"] is not None:
                options_proj["HIP_steps"] = steps["genfun_HIP"](
                    *options_proj["genarg_HIP"]
                )

            # Projection loops
            if proj == "oneHIP":
                projection_with_storage = one_step_HIP_with_storage
            elif (
                proj == "Dykstra"
            ):  # Dykstra from first CP estimator, not from LS estimator.
                projection_with_storage = Dykstra_with_storage
            elif proj == "pureHIP":
                projection_with_storage = pure_HIP_with_storage
            elif proj == "Alternate":
                projection_with_storage = alternate_projections_with_storage
            elif proj == "HIPswitch":
                projection_with_storage = (
                    hyperplane_intersection_projection_switch_with_storage
                )
            elif proj == "HIPrec":
                projection_with_storage = (
                    hyperplane_intersection_projection_recall_with_storage
                )
            else:
                raise ValueError(
                    'unknown projection. Should be "oneHIP", "pureHIP", "Alternate", "HIPswitch" or "Dykstra"'
                )

            rho, dt, comp_time, iterations = projection_with_storage(
                rhoCP,
                group,
                true_Choi,
                **options_proj,
                least_ev_x_dim2_tol=least_ev_x_dim2_tol,
                all_dists=all_dists,
                with_evs=with_evs,
                dist_L2=dist_L2,
                save_intermediate=save_intermediate,
            )

            time_exp += comp_time
            #
            #
            # End of estimation, with final adjustments

            table_summary.row[f"{proj}_number_of_iterations"] = iterations
            end = group.end.row
            store_distances_all(
                end,
                rho - true_Choi,
                prefix="TPfinal_",
                with_evs=True,
                error_array=group.TPfinal_evs_error,
                summary_row=table_summary.row,
                summary_prefix=f"{proj}_TPfinal_",
            )
            store_fidelity(end, rho, true_Choi, prefix="TPfinal_")
            end["TP_proj_time"] = dt
            time_exp += dt

            t0 = time.perf_counter()
            rho, least_ev = final_CPTP_by_mixing(rho, full_output=True)
            t1 = time.perf_counter()
            end["final_proj_time"] = t1 - t0
            end["TPfinal_least_ev"] = least_ev
            time_exp += t1 - t0

            store_distances_all(
                end,
                rho - true_Choi,
                prefix="final_",
                error_array=group.final_evs_error,
                with_evs=True,
                summary_row=table_summary.row,
                summary_prefix=f"{proj}_final_",
            )
            store_fidelity(end, rho, true_Choi, prefix="final_")

            end.append()
            group.end.flush()
            if keep_key_channels:
                fileh.create_array(
                    group,
                    "final_rho_hat",
                    rho,
                    "Final estimator of the Choi matrix of the channel",
                )
            if keep_main_eigenvectors != 0:
                main_eigvals, main_eigvecs = SL.eigh(
                    rho,
                    subset_by_index=[dim ** 2 - keep_main_eigenvectors, dim ** 2 - 1],
                )
                fileh.create_array(
                    group,
                    "main_eigvals_final",
                    main_eigvals,
                    "Main eigenvalues of the final estimator.",
                )
                fileh.create_array(
                    group,
                    "main_eigvecs_final",
                    main_eigvecs,
                    "Main eigenvectors of the final estimator.",
                )

        table_summary.row["time_exp"] = time_exp
        compute_time += time_exp
        table_summary.row.append()
        table_summary.flush()

    table_summary.attrs.total_computation_time = compute_time
    t_final = time.perf_counter()
    table_summary.attrs.wall_time = t_final - t_init
    table_summary.attrs.total_logging_time = t_final - t_init - compute_time

    # END FUNCTION :D


def generate_simulations(
    filename,
    dim,
    channel,
    scenario="Pauli",
    cycles=1,
    projections=["HIPswitch"],
    repetitions=1,
    options_proj={},
    default_options_proj={
        "maxiter": 300,
        "HIP_to_alt_switch": "first",
        "missing_w": 3,
        "min_part": 0.1,
        "HIP_steps": 10,
        "alt_steps": 4,
        "alt_to_HIP_switch": "cos",
        "min_cos": 0.99,
        "max_mem_w": 30,
        "genarg_alt": (1, 3, 20),
        "genarg_HIP": (5,),
    },
    steps={"genfun_alt": None, "genfun_HIP": None},
    depo_tol=1e-3,
    depo_rtol=1e-1,
    first_CP_threshold_least_ev=True,
    all_dists=False,
    dist_L2=True,
    with_evs=False,
    keep_key_channels=False,
    keep_main_eigenvectors=0,
    save_intermediate=False,
    bigmem=False,
    random_state=None,
):
    """
    filename: name of the file where the results are stored.
    
    dim: dimension of the underlying Hilbert space.
    
    scenario: 'Pauli' is Pauli without ancilla; dim must be $2**k$, $k$ integer.
              'MUBs' is mutually unbiased bases without ancilla. dim must be a prime other than 2.
    
    channel: channel to be estimated, defined by its Kraus operators. Array shape (rank, dim, dim).
    
    projection: list of projections used,
                'HIPswitch' is hyperplane_intersection_projection_switch.
                'HIPrec' is hyperplane_intersection_projection_recall.
                'Dykstra' is Dykstra_projection (starting with CP estimator instead of LS  estimator).
                'Alternate' is alternate projections (starting with CP).
                'pureHIP' is pure_HIP. 
                'oneHIP' is one_step_HIP.
                If empty, stops after the first projection on CP maps, without trying to ensure it is trace-preserving.
    
    repetitions: number of repetitions of the simulation.    
    
    options_proj: options passed to the projection algorithm. Reasonable default
        choices automatically loaded for all non-provided ones for all projection 
        choices and switch options in HIPSwitch.
    
    steps: other options passed to the projection algorithm, that may not be picklable. In particular, to use
        HIPSwitch with 'counter' and a generator, provide the generator-building funciton in 'genfun_alt' or
        'genfun_HIP', and the arguments to that function in 'genarg_alt' or 'genarg_HIP', which are in options_proj.
    
    depo_tol: Maximum added mistake to half the $L^1$ distance when mixing the depolarizing channel in the
              end. 
              When either (as opposed to both) depo_tol or depo_rtol is attained, stops.

    depo_rtol: Maximum relative added mistake to half the $L^1$ distance when mixing the depolarizing channel 
               in the end. Namely if the mistake before the mixing is $\epsilon$, the mistake after the mixing
               is at most $(1 + depo_rtol) \epsilon$.
               When either (as opposed to both) depo_tol or depo_rtol is attained, stops.
    
    first_CP_threshold_least_ev: If True, the first projection on CP is used after thresholding by minus the 
              least eigenvalue, if that can be done without changing any eigenvalue by more than this 
              threshold.
    
    all_dists: If True, saves not only the $L^2$ distance, but also the $L^1$ and $L^\infty$ distance between 
               estimator and real channel at each step. DOUBLES THE COMPUTATION TIME.

    dist_L2:   If True, saves the $L^2$ distance between estimator and real channel at each step.
    
    with_evs:  If True, saves the eigenvalues of the estimator at each step.
    
    keep_key_channels: If True, saves the LS estimator, the first CP estimator, and
       the final estimator. Big if the underlying dimension is big ($O(d^4)$).
                     
    keep_main_eigenvectors: If nonzero, keeps that number of highest eigenvalues and
        associated eigenvectors for the true channel, the LS, the CP, and the final estimators.

    save_intermediate: If True, saves the estimator at every step. Can get very big!

    bigmem: If True, slightly less aggressive on memory conservation.
    
    random_state: initialisation of the random generator. WARNING: Global numpy 
        RandomState is reinitialized if the option is not None.
    """

    for key in default_options_proj:
        if key not in options_proj:
            options_proj[key] = default_options_proj[key]

    if filename[-3:] != ".h5":
        filename += ".h5"

    assert not Path(filename).exists(), "File already exists."

    try:
        fileh = tables.open_file(filename, mode="w")

        _generate_simulations(
            fileh,
            dim,
            channel,
            scenario,
            cycles,
            projections,
            repetitions,
            options_proj,
            steps,
            depo_tol,
            depo_rtol,
            first_CP_threshold_least_ev,
            all_dists,
            dist_L2,
            with_evs,
            keep_key_channels,
            keep_main_eigenvectors,
            save_intermediate,
            bigmem,
            random_state,
        )

    finally:
        fileh.close()


def retrieve_main(fileh_name, exp_number=0, proj_type="HIPswitch", full_matrices=False):
    """
    Retrieves in a dictionary some of the information from a file created by 
    generate_simulations.

    Only one experiment repetition and projection type at a time. The experiment
    number is exp_number, 0 by default.
    The projection type is one of 'HIPswitch' (default), 'HIPrec', 'oneHIP', 'pureHIP',
    'Alternate' or 'Dykstra'.

    The output dictionary has the following entries:
    * sample_size : the number of samples really taken.
    * time_proj: the time taken by the projection, without the data generation and 
    without the logging.
    * 'dist_L2': vector of L2 distances between estimator and real channel. First
    element is with respect to the LS estimator, second element is the first 
    projection on CP, then each successive pair of elements are respectively on
    TP and CP. Last two elements are the distance of the final estimator pre-mixing,
    and that of the final estimator.
    * 'dist_L1': If available, same structure as dist_L2. Otherwise only contains 
    the  distances from the LS estimator, the first on CP, the final pre-mixing and 
    the final estimator ones.
    * 'dist_Linfty': same as dist_L1.
    * 'least_ev': Least eigenvalue of the TP estimators, with first element being
    the LS estimator, and the last being the final pre-mixing one.
    * 'evs_error': Eigenvalues of the difference between estimator and true channel.
    The included estimators are either all of them, or just the four: LS, first CP,
    final pre-mixing, final (like for 'dist_L1')
    * if full_matrices=True and if they are included:
    ** 'true_channel' : the Choi matrix of the true channel
    ** 'intermediate': the intermediate TP estimators, as an 3d array; first
           dimension is the number of the TP estimator.
    ** 'final': the final estimator
    ** 'LS'   : least_square estimator
    ** 'CP'   : the first CP estimator
    """

    data = {}
    with tables.open_file(fileh_name, "r") as fileh:
        assert (
            f"exp_repet{exp_number}" in fileh.root
        ), "No such experiment number exp_number. Try 0."
        exp_rep = fileh.get_node(f"/exp_repet{exp_number}")
        assert f"projection_{proj_type}" in exp_rep, "No such projection was used."
        proj = fileh.get_node(f"/exp_repet{exp_number}/projection_{proj_type}")

        data["sample_size"] = fileh.root.summary.cols.sample_size[exp_number]
        data["time_proj"] = (
            proj.loops.attrs.computation_time
            + proj.end.cols.TP_proj_time[0]
            + proj.end.cols.final_proj_time[0]
            + exp_rep.initialisation.cols.CP_proj_time[0]
        )
        data["dist_L2"] = N.array([exp_rep.initialisation.cols.LS_dist_L2[0]])
        data["dist_L2"] = N.r_[
            data["dist_L2"], exp_rep.initialisation.cols.CP_dist_L2[0]
        ]
        data["dist_L1"] = N.array([exp_rep.initialisation.cols.LS_dist_L1[0]])
        data["dist_L1"] = N.r_[
            data["dist_L1"], exp_rep.initialisation.cols.CP_dist_L1[0]
        ]
        data["dist_Linfty"] = N.array([exp_rep.initialisation.cols.LS_dist_Linfty[0]])
        data["dist_Linfty"] = N.r_[
            data["dist_Linfty"], exp_rep.initialisation.cols.CP_dist_Linfty[0]
        ]
        data["least_ev"] = N.array([exp_rep.initialisation.cols.LS_least_ev[0]])
        data["evs_error"] = N.r_[exp_rep.LS_evs_error, exp_rep.CP_evs_error]

        data["dist_L2"] = N.r_[
            data["dist_L2"],
            N.c_[proj.loops.cols.TP_dist_L2, proj.loops.cols.CP_dist_L2].ravel(),
        ]
        data["least_ev"] = N.r_[data["least_ev"], proj.loops.cols.TP_least_ev]
        if "TP_dist_L1" in proj.loops.colnames:
            data["dist_L1"] = N.r_[
                data["dist_L1"],
                N.c_[proj.loops.cols.TP_dist_L1, proj.loops.cols.CP_dist_L1].ravel(),
            ]
            data["dist_Linfty"] = N.r_[
                data["dist_Linfty"],
                N.c_[
                    proj.loops.cols.TP_dist_Linfty, proj.loops.cols.CP_dist_Linfty
                ].ravel(),
            ]
        if "TP_evs_error" in proj:
            dimC = len(data["evs_error"][0])
            data["evs_error"] = N.r_[
                data["evs_error"],
                N.c_[proj.TP_evs_error, proj.CP_evs_error].reshape(-1, dimC),
            ]

        data["dist_L2"] = N.r_[
            data["dist_L2"],
            proj.end.cols.TPfinal_dist_L2[0],
            proj.end.cols.final_dist_L2[0],
        ]
        data["dist_L1"] = N.r_[
            data["dist_L1"],
            proj.end.cols.TPfinal_dist_L1[0],
            proj.end.cols.final_dist_L1[0],
        ]
        data["dist_Linfty"] = N.r_[
            data["dist_Linfty"],
            proj.end.cols.TPfinal_dist_Linfty[0],
            proj.end.cols.final_dist_Linfty[0],
        ]
        if (
            "TPfinal_least_ev" in proj.end.colnames
        ):  # Older versions did not have that entry
            data["least_ev"] = N.r_[data["least_ev"], proj.end.cols.TPfinal_least_ev]
        else:
            print(
                "Warning: in this version of the file, the least eigenvalue of the pre-mixing final estimator is not "
                + "stored. Hence the last value in 'least_ev' is that of the last TP estimator in the loop."
            )
        data["evs_error"] = N.r_[
            data["evs_error"], proj.TPfinal_evs_error, proj.final_evs_error
        ]

        if full_matrices:
            data["true"] = Choi(fileh.root.kraus_true_channel[:])
            if "LS_estimator" in exp_rep:
                data["LS"] = exp_rep.LS_estimator[:]
                data["CP"] = exp_rep.CP_estimator[:]
            if "rhoTP" in proj:
                data["intermediate"] = proj.rhoTP[:]
            if "final_rho_hat" in proj:
                data["final"] = proj.final_rho_hat[:]

        return data
