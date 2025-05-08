
# ladder_s21_gpu.py
import numpy as np
import torch
import torch.nn as nn

class LadderS21(nn.Module):
    """
    Fixed network for a 5-section series-first LC ladder.
    Inputs
        params : (..., 10) real tensor
                 order = [C1,L1,C2,L2,C3,L3,C4,L4,C5,L5]
    Output
        S21    : (..., 1000) complex tensor  (f = 0 … 5 GHz)
    """

    def __init__(self, z0=50.0, n_pts=1000, f_max=5e9):
        super().__init__()
        f = torch.linspace(0.0, f_max, n_pts)          # Hz
        self.register_buffer("freq", f)                # shape (F,)
        self.z0 = float(z0)

    def forward(self, params):
        if params.shape[-1] != 10:
            raise ValueError("params last dim must be 10")

        # promote to complex for math
        params = params.to(torch.float32)

        C = params[..., 0::2]          # (..., 5)
        L = params[..., 1::2]          # (..., 5)

        # angular freq, broadcast to match batch dims
        F = self.freq.numel()
        batch_shape = params.shape[:-1]
        w = 2.0 * np.pi * self.freq       # (F,)
        w = w.view((1,) * len(batch_shape) + (F,)) + 1e-30  # avoid div0

        # convenience tensors for results
        dtype = torch.cfloat
        device = params.device
        abcd = torch.eye(2, dtype=dtype, device=device)
        abcd = abcd.expand(*batch_shape, F, 2, 2).clone()   # (...,F,2,2)

        for i in range(5):
            Ci = C[..., i].unsqueeze(-1)    # (...,1)
            Li = L[..., i].unsqueeze(-1)

            # series combo impedance
            Zs = 1j * w * Li + 1.0 / (1j * w * Ci)
            # parallel combo admittance
            Yp = 1j * w * Ci + 1.0 / (1j * w * Li)

            # element ABCD
            elem = torch.zeros_like(abcd)
            elem[..., 0, 0] = 1.0
            elem[..., 1, 1] = 1.0

            if i % 2 == 0:                # series element (0,2,4)
                elem[..., 0, 1] = Zs
            else:                         # shunt element (1,3)
                elem[..., 1, 0] = Yp

            abcd = torch.matmul(abcd, elem)   # running product

        a = abcd[..., 0, 0]
        b = abcd[..., 0, 1]
        c = abcd[..., 1, 0]
        d = abcd[..., 1, 1]

        denom = a + b / self.z0 + c * self.z0 + d
        s21 = 2.0 / denom                  # (...,F)

        s21_mag_db = 20.0 * torch.log10(torch.abs(s21) + 1e-30)
        return s21_mag_db





# def circuit2s(crct_obj: EE.Circuit, f_pts: np.ndarray) -> np.ndarray:
#     """
#     Compute S-parameters of a ladder LC network described by `crct`.

#     Parameters
#     ----------
#     crct : np.ndarray
#         2-D array with 5 rows as described in the module docstring.
#     f_pts : array_like
#         Frequency points in Hz.

#     Returns
#     -------
#     s_param_out : np.ndarray
#         S-parameters with shape (len(f_pts), 2, 2).
#     """
    
#     crct_obj.
    
    
#     crct = np.asarray(crct)
#     f_pts = cir

#     ORD = crct.shape[1] - 1  # exclude load column

#     RES = 0
#     IND = 1
#     CAP = 2
#     COMPS_SERIES = 3
#     ELEM_SERIES = 4

#     # allocate ABCD array: (n_freq, 2, 2)
#     ABCD = np.empty((len(f_pts), 2, 2), dtype=complex)

#     for fi, freq in enumerate(f_pts):
#         w = 2.0 * np.pi * freq

#         # running ABCD for this frequency
#         abcd_run = np.eye(2, dtype=complex)

#         for k in range(ORD):
#             if crct[COMPS_SERIES, k] == 1:
#                 # L and C in series
#                 z_temp = 1j * w * crct[IND, k] + 1.0 / (1j * w * crct[CAP, k])
#                 y_temp = 1.0 / z_temp
#             else:
#                 # L and C in parallel
#                 y_temp = 1.0 / (1j * w * crct[IND, k]) + 1j * w * crct[CAP, k]
#                 z_temp = 1.0 / y_temp

#             if crct[ELEM_SERIES, k] == 1:
#                 abcd_elem = np.array([[1.0, z_temp],
#                                       [0.0, 1.0]], dtype=complex)
#             else:
#                 abcd_elem = np.array([[1.0, 0.0],
#                                       [y_temp, 1.0]], dtype=complex)

#             abcd_run = abcd_run @ abcd_elem

#         ABCD[fi] = abcd_run

#     # convert to S-parameters
#     s_param_out = abcd2s(ABCD)
#     return s_param_out









# """
# ASCII-only implementation of circuit2s() from MATLAB.

# Assumptions:
# - Source and load reference impedance are both 50 ohms.
# - `crct` is a 2-D array (rows = 5, columns = elements + 1)
#   Row layout (1-indexed in MATLAB, 0-indexed here):
#     0: Resistance (not used in this translation)
#     1: Inductance values (H)
#     2: Capacitance values (F)
#     3: COMPS_SERIES flag (1 = L and C are in series, 0 = in parallel)
#     4: ELEM_SERIES flag  (1 = series element, 0 = shunt element)
# - The last column in `crct` is the load; only the first `ORD` columns
#   (where ORD = num_cols - 1) are processed.

# Functions:
#     circuit2s(crct, f_pts)
# """


# Z0 = 50.0  # reference impedance in ohms


# def abcd2s(abcd):
#     """
#     Convert ABCD parameters (shape (..., 2, 2)) to S-parameters.

#     Parameters
#     ----------
#     abcd : np.ndarray
#         Array of ABCD matrices. The last two dimensions must be 2x2.

#     Returns
#     -------
#     s : np.ndarray
#         Array of S-parameter matrices with the same leading dimensions.
#     """
#     a = abcd[..., 0, 0]
#     b = abcd[..., 0, 1]
#     c = abcd[..., 1, 0]
#     d = abcd[..., 1, 1]

#     denom = a + b / Z0 + c * Z0 + d
#     s11 = (a + b / Z0 - c * Z0 - d) / denom
#     s21 = 2.0 / denom
#     s12 = 2.0 * (a * d - b * c) / denom
#     s22 = (-a + b / Z0 - c * Z0 + d) / denom

#     # assemble into (..., 2, 2)
#     shape = s11.shape + (2, 2)
#     s = np.empty(shape, dtype=complex)
#     s[..., 0, 0] = s11
#     s[..., 0, 1] = s12
#     s[..., 1, 0] = s21
#     s[..., 1, 1] = s22
#     return s



