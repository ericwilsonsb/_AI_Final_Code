



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
import torch

# import importlib
import sys
sys.path.append('../EE_utils/')
import electrical as EE # type: ignore
# importlib.reload(EE)
import electrical_sim as EE_sim # type: ignore
# importlib.reload(EE_sim)



def gen_cheby_data(num_filters: int = 1000, num_f_pts: int = 1000, comp_var: float = np.nan, device: str = "cuda") -> tuple[np.ndarray, np.ndarray]:

    # HyperCube Sampling
    d = 3               # number of dimensions (center_freq, frac_BW, ripple_dB)
    m = num_filters     # total number of samples

    sampler = qmc.LatinHypercube(d=d)
    unit_sample = sampler.random(n=m)

                    # center_freq, frac_BW, ripple_dB
    lower_bnd = np.array([1e9, 0.1, 0.1])
    upper_bnd = np.array([4e9,  3, 2.99])
    scaled_sample = qmc.scale(unit_sample, lower_bnd, upper_bnd)   # same shape



    tensor_designs = torch.empty((m, 10))


    # Generate Clean Filters
    for r in range(m):

        ripple_dB = scaled_sample[r, 2]
        freq_center = scaled_sample[r, 0]
        frac_BW = scaled_sample[r, 1]

        tensor_designs[r,:] = EE.generate_BP( filter_style="chebyshev", order=5, ripple_dB=ripple_dB, freq_center=freq_center, frac_BW=frac_BW ).elements_2_tensor()


    # device = "cuda"
    # device = "cpu"

    filter_simulator = EE_sim.LadderS21(n_pts=num_f_pts).to(device)   # use "cpu" for CPU
    tensor_designs_CUDA = tensor_designs.to(device)

    data_S21_dB_CUDA = filter_simulator( tensor_designs_CUDA )

    s21_db = data_S21_dB_CUDA.to("cpu").numpy()

    return tensor_designs.numpy(), s21_db








def gen_rand_data(num_filters: int = 1000, num_f_pts: int = 1000, comp_val_min: float = 1e-14, comp_val_max: float = 1e-6, device: str = "cuda") -> tuple[np.ndarray, np.ndarray]:

    # HyperCube Sampling
    d = 10              # number of dimensions (C1, L1, C2, L2, ... C5, L5)
    m = num_filters     # total number of samples

    sampler = qmc.LatinHypercube(d=d)
    unit_sample = sampler.random(n=m)

                    # center_freq, frac_BW, ripple_dB
    lower_bnd = comp_val_min * np.ones(d)
    upper_bnd = comp_val_max * np.ones(d)
    scaled_sample = qmc.scale(unit_sample, lower_bnd, upper_bnd)   # same shape



    # Generate Random Valued Filters
    tensor_designs = torch.from_numpy( scaled_sample )


    # device = "cuda"
    # device = "cpu"

    filter_simulator = EE_sim.LadderS21(n_pts=num_f_pts).to(device)   # use "cpu" for CPU
    tensor_designs_CUDA = tensor_designs.to(device)

    data_S21_dB_CUDA = filter_simulator( tensor_designs_CUDA )

    s21_db = data_S21_dB_CUDA.to("cpu").numpy()

    return tensor_designs.numpy(), s21_db