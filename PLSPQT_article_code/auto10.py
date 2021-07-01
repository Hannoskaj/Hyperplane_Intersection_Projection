import experiment_generation as EG 
import channels as Ch
import numpy as np
import os




try:
    os.mkdir('Jonas_experiment_7_qubits')
except FileExistsError:
    pass

dim= 2**7

for ss, ssl in [(1e10, '1e10')]:
    EG.generate_simulations(f'Jonas_experiment_7_qubits/vignette_7_qubits_{ssl}.h5', dim,
        Ch.add_disentanglement_noise(Ch.QFTKraus(dim), level=.25), scenario='Pauli', 
        cycles= ss / 18**7, projections=['HIPswitch'], repetitions=1,
        options_proj={'maxiter':200, 'HIP_to_alt_switch':'first', 'alt_to_HIP_switch':'cos', 
                'min_cos':.99, 'max_mem_w':30},
        depo_tol=5e-3, depo_rtol=1e-10, first_CP_threshold_least_ev=True, all_dists=False,
        keep_key_channels=False, keep_main_eigenvectors=2, save_intermediate=False,
        random_state=1234567890)

