# -*- coding: utf-8 -*-


import numpy as np
import torch

class Circuit:

    def __init__(self, element_arr=np.nan):

        self.elements = element_arr  # circuit elements
        # top row = inductors (H)
        # bottom row = capacitors (F)
        # collumns index 0, 2, 4 are series elements, and series in ladder,
            # else shunt elements, shunt in ladder

        self.s_params = np.nan  # s-parameters
        
        # freq data (Hz)
        self.f_start = 0.1e9
        self.f_end = 5e9
        self.f_step = 5e6

        # optional info
        self.filter_type = ""
        self.order = np.nan
        self.freq_center = np.nan
        self.ripple_dB = np.nan
        self.frac_BW = np.nan
        self.comp_var = np.nan


    def __str__(self):
        """
        Prettyprint a 2D array with centred column headers.
        """
        if np.isnan( self.elements ).all():
            return "Filter Elements NOT DEFINED!!!"

        fmt = "{:>3.3g}"    # Number formatting

        n_rows, n_cols = self.elements.shape

        col_labels = [f"C{c+1}" for c in range(n_cols)]
        row_labels = ["L", "C"]

        # preformat numeric cells to compute widths
        cell_strs = [[fmt.format(self.elements[r, c]) for c in range(n_cols)] for r in range(n_rows)]

        col_widths = [
            max(len(col_labels[c]), max(len(cell_strs[r][c]) for r in range(n_rows)))
            for c in range(n_cols)
        ]
        row_lab_w = max(len(max(row_labels, key=len)), 2)

        # centred header
        header = " " * (row_lab_w + 3) + " ".join(
            col_labels[c].center(col_widths[c]) for c in range(n_cols)
        )
        separator = " " * (row_lab_w + 3) + " ".join("-" * col_widths[c] for c in range(n_cols))

        # data rows
        lines = ["-------------------------------------------------", header, separator]
        for r in range(n_rows):
            line = (
                row_labels[r].ljust(row_lab_w)
                + " | "
                + " ".join(cell_strs[r][c].rjust(col_widths[c]) for c in range(n_cols))
            )
            lines.append(line)

        lines.append(f"\nfilter_type = {self.filter_type}")
        lines.append(f"order = {self.order}")
        lines.append(f"freq_center = {self.freq_center}")
        lines.append(f"frac_BW = {self.frac_BW}")
        lines.append(f"comp_var = {self.comp_var}")
        lines.append("-------------------------------------------------\n")

        return "\n".join(lines)



    # returns np list of frequencies (Hz)
    def compute_freq(self):
        # return True
        return np.arange( int(self.f_start), int(self.f_end + self.f_step), int(self.f_step) )
    



    def elements_2_tensor(self) -> torch.tensor:

        params = torch.from_numpy(
                        self.elements[[1, 0]]  # swap rows: capacitors, inductors
                        .T                        # transpose to shape (5,2)
                        .reshape(-1)              # flatten to 10‑element vector
                        ).to(torch.float32)       # LadderS21 expects float32

        return params


    

    
    







def generate_BP( filter_style: str, order: int, ripple_dB: float, freq_center: float, frac_BW: float, comp_var: float=np.nan ) -> Circuit:

    # Indexing vals
    IND = 0
    CAP = 1

    # Electrical vals
    z0 = 50 #ohms
    w_0 = 2 * np.pi * freq_center

    # array for comp. vals
    comp_vals = np.full( (2,order), np.nan )

    # Gernate G-Coefs
    g_r = generate_norm_LP_coef(filter_style=filter_style, n_max=order, ripple_dB=ripple_dB)[order-1, :]

    # Scale Series Elements (even collumn index)
    for col in np.arange(0, order, 2):
        comp_vals[IND, col] = z0*g_r[col] / ( w_0 * frac_BW )       # scale INDUCTOR
        comp_vals[CAP, col] = frac_BW / ( z0 * w_0 * g_r[col] )     # scale CAPACITOR

    # Scale Shunt Elements (odd collumn index)
    for col in np.arange(1, order, 2):
        comp_vals[IND, col] = z0 * frac_BW / ( w_0 * g_r[col] )     # scale INDUCTOR
        comp_vals[CAP, col] = g_r[col] / ( z0 * frac_BW * w_0 )     # scale CAPACITOR

    # Apply Random Component Variance
    if( not np.isnan(comp_var) ):
        comp_variances = np.random.uniform( low=1-comp_var, high=1+comp_var, size=(2, order) )
        comp_vals = comp_variances * comp_vals

    # Create Circuit
    BP_filter = Circuit( element_arr=comp_vals )

    # Save Useful Data
    BP_filter.filter_type = filter_style
    BP_filter.oder = order
    BP_filter.freq_center = freq_center
    BP_filter.ripple_dB = ripple_dB
    BP_filter.frac_BW = frac_BW
    BP_filter.comp_var = comp_var

    return BP_filter
    
        




    











def generate_norm_LP_coef(filter_style: str, n_max: int, ripple_dB: float | None = None) -> np.ndarray:
    """
    Generate normalized lowpass prototype element coefficients (gvalues).

    ( I wrote this script in MATLAB and had AI convert it to python )

    Parameters
    ----------
    filter_style : str
        Either "butterworth" or "chebyshev" (caseinsensitive).
    n_max : int
        Maximum filter order.
    ripple_dB : float, optional
        Passband ripple in dB (required for Chebyshev).

    Returns
    -------
    g_r : np.ndarray
        Matrix of prototype coefficients, shape = (n_max, n_max + 1).

    Raises
    ------
    ValueError
        If the filter style is unsupported or if ripple_dB is
        missing when Chebyshev is selected.
    """
    style = filter_style.lower()

    if style == "butterworth":
        return butterworth_coef(n_max)

    elif style == "chebyshev":
        if ripple_dB is None:
            raise ValueError("`ripple_dB` must be provided for a Chebyshev filter.")
        return chebyshev_coef(n_max, ripple_dB)

    else:
        raise ValueError(f"Unsupported filter style: {filter_style!r}")





def butterworth_coef(n_max: int) -> np.ndarray:
    """
    Generate Butterworth prototype element coefficients.

    ( I wrote this script in MATLAB and had AI convert it to python )

    Parameters
    ----------
    n_max : int
        Maximum filter order (number of rows).

    Returns
    -------
    g_r : np.ndarray  (shape = (n_max, n_max + 1))
        Matrix whose ith row (order = i+1) contains
        g  g and a trailing 1, all others set to NaN.
    """
    # initialise with NaNs
    g_r = np.full((n_max, n_max + 1), np.nan)

    # loop over desired order (1based in MATLAB  0based here)
    for n_row in range(1, n_max + 1):
        r = np.arange(1, n_row + 1)                          # 1  n_row
        g_vals = 2 * np.sin((2 * r - 1) * np.pi / (2 * n_row))
        g_r[n_row - 1, :n_row] = g_vals                      # fill g  g
        g_r[n_row - 1, n_row] = 1.0                          # trailing 1

    return g_r








def chebyshev_coef(n_max: int, ripple_dB: float) -> np.ndarray:
    """
    Calculate the elementvalue (gvalue) table for Chebyshev filters.

    ( I wrote this script in MATLAB and had AI convert it to python )

    Parameters
    ----------
    n_max : int
        Maximum prototype order you want in the table.
    ripple_dB : float
        Passband ripple in dB (3dB  return loss form, >3dB  ripple form).

    Returns
    -------
    g_r : np.ndarray, shape (n_max, n_max+3)
        Row i (0based) contains the gvalues for an i+1order prototype.
        The last extra column stores an alternate load value when the first
        element is a shunt capacitor instead of a series inductor (even order).
    """
    # 
    #  Ripple handling (match MATLAB logic)
    # 
    if ripple_dB > 3:                                   # return loss region
        L_R_lin = 10 ** (-ripple_dB / 10)
        LA_dB_ripple = -10 * np.log10(1 - L_R_lin)
    else:                                               # direct ripple spec
        _ = 10 * np.log10(1 - 10 ** (-ripple_dB / 10))  # stored as return_ripple_dB in MATLAB
        LA_dB_ripple = ripple_dB

    epsilon = np.sqrt(10 ** (LA_dB_ripple / 10) - 1)    # single  value

    # gvalue table (extra two cols match MATLAB comment)
    g_r = np.full((n_max, n_max + 3), np.nan, dtype=float)

    # 
    #  Populate each order
    # 
    for n_row in range(1, n_max + 1):  # MATLAB rows 1n_max
        eta = np.sinh((1 / n_row) * np.arcsinh(1 / epsilon))
        g_r[n_row - 1, 0] = (2 / eta) * np.sin(np.pi / (2 * n_row))

        # inner g_r values
        for r in range(1, n_row):
            A = 4 * np.sin(((2 * r - 1) * np.pi) / (2 * n_row))
            B = np.sin(((2 * r + 1) * np.pi) / (2 * n_row))
            C = eta ** 2 + np.sin(r * np.pi / n_row) ** 2
            g_r[n_row - 1, r] = A * B / (C * g_r[n_row - 1, r - 1])

        # load term(s)
        if n_row % 2 == 1:                               # odd order  load = 1
            g_r[n_row - 1, n_row] = 1.0
        else:                                            # even order
            g_tmp = (epsilon + np.sqrt(1 + epsilon ** 2)) ** 2
            g_r[n_row - 1, n_row] = g_tmp               # seriesL first
            g_r[n_row - 1, n_max + 2] = 1 / g_tmp       # shuntC first (extra col)

    return g_r





