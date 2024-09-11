import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from scipy.integrate import cumtrapz
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import brentq
from scipy.sparse import csc_matrix
from scipy.stats import multivariate_normal
from tqdm import tqdm


def pol2cart(rho, theta):
    x = np.multiply(rho, np.cos(theta))
    y = np.multiply(rho, np.sin(theta))
    return x, y


def cart2pol(x, y):
    phi = np.arctan2(y, x)
    rho = np.sqrt(x**2 + y**2)
    return phi, rho


def compute_count_difference(E2v, rho_array, count_rgc, ind_rho_corrected, sector_dtheta):
    count_rf = compute_count_rf(E2v, rho_array, sector_dtheta)

    difference = count_rf[ind_rho_corrected] - count_rgc[ind_rho_corrected]
    return difference


def compute_count_rf(E2v, rho_array, sector_dtheta):
    Rv = 0.011785
    Ro = 0.008333
    a = 1.004
    b = -0.007209
    c = 0.001694
    d = -0.00003765

    r_ret_sphere = 11.459  # [mm]
    deg2mm = 2 * np.pi * r_ret_sphere / 360
    eccentricity = rho_array / deg2mm

    k = 1 + np.power((a + b * eccentricity + c * eccentricity ** 2 + d * eccentricity ** 3), -2)
    d_rf_degree = k * (1.12 + 0.0273 * eccentricity) / (
                1.115 * ((Rv * (1 + eccentricity / E2v)) ** 2 - (Ro * (1 + eccentricity / 20)) ** 2))
    d_rf = d_rf_degree / deg2mm ** 2
    count_rf = cumtrapz(d_rf * sector_dtheta * rho_array, rho_array)

    return count_rf


def kernel_sigma(eccentricity):
    return 10 * 0.0263 * (1 + 3.15 / (7.99 / (1 + 0.29 * eccentricity + 0.000012 * eccentricity ** 2))) / (
                2 * np.sqrt(2 * np.log(2)))  # E in degrees


def kernel_func(locs, locs_rgc_rf, dxdy, deg2mm):
    dx = dxdy[0]
    dy = dxdy[1]
    mm2deg = 1 / deg2mm
    locs_rgc_rf_rho = np.sqrt(np.sum(locs_rgc_rf ** 2, axis=0))
    sigma = kernel_sigma(locs_rgc_rf_rho * mm2deg) * deg2mm
    cov = sigma ** 2
    kernel = 300 * dx * dy * multivariate_normal.pdf(locs, mean=locs_rgc_rf, cov=cov)
    return kernel


def sample_rgc_population(
        n_fibers_theor=10000, 
        radius_retina=10, 
        radius_optic_nerve=2.5, 
        n_interp_rho=1000, 
        n_interp_theta=1000, 
        density_file_path='curciodensity.xlsx'
    ):
    """
    Samples the location of RGCs in the three domains: visual field,
    retina, and optic nerve section.

    Parameters
    ----------
    n_fibers_theor :
        Number of RGCs to be sampled.
    radius_retina :
        Radius of the retina [mm].
    radius_optic_nerve :
        Radius of the optic nerve [mm].
    n_interp_rho :
        Number of steps in the radial discretization of RGC soma density.
    n_interp_theta :
        Number of steps in the angular discretization of RGC soma density.
    density_file_path :
        Path to the XLSX file with RGC soma density measured by Curcio et al.

    Returns
    -------

    """

    # Load Curcio's data
    T = pd.read_excel(density_file_path)
    eccentricity = T['eccentricity'].to_numpy()
    d_temp = T['temporal_mean'].to_numpy()
    d_nas = T['nasal_mean'].to_numpy()
    d_sup = T['superior_mean'].to_numpy()
    d_inf = T['inferior_mean'].to_numpy()

    # remove Curcio's optic disc (null element in their data)
    idx_null = 17
    d_nas = np.delete(d_nas, idx_null)

    ecc_nas = eccentricity
    ecc_nas = np.delete(ecc_nas, idx_null)

    # == Interpolation
    rho = np.linspace(0, radius_retina, n_interp_rho)

    # interpolate radially
    cs_nas = CubicSpline(ecc_nas, d_nas)
    cs_temp = CubicSpline(eccentricity, d_temp)
    cs_sup = CubicSpline(eccentricity, d_sup)
    cs_inf = CubicSpline(eccentricity, d_inf)

    d_nas_interp = cs_nas(rho)
    d_temp_interp = cs_temp(rho)
    d_sup_interp = cs_sup(rho)
    d_inf_interp = cs_inf(rho)

    # interpolate angularly
    theta = np.linspace(0, 2 * np.pi, n_interp_theta)
    d_interp = np.zeros((n_interp_theta, n_interp_rho))
    for i in range(1, n_interp_rho):
        pts = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]) * rho[i]
        density = np.array(
            [
                d_nas_interp[i],
                d_sup_interp[i],
                d_temp_interp[i],
                d_inf_interp[i],
                d_nas_interp[i]
            ]
        )

        cs_theta = CubicSpline(pts, density)
        d_interp[:, i] = cs_theta(theta * rho[i])

    [Rho, Theta] = np.meshgrid(rho, theta)
    [X, Y] = pol2cart(Rho, Theta)

    # optic disc ellipsis
    #
    # (x - x_c)**2      (y - 0)**2
    # --------     +    -------    = 1
    #   a_h**2           a_v**2
    #
    a_optic_disc = np.array([1.88, 1.77]) / 2
    x_optic_disc = eccentricity[idx_null]
    idx_optic_disc = np.logical_and(
        np.abs(X - x_optic_disc) < a_optic_disc[0],
        np.abs(Y) < a_optic_disc[1] * np.sqrt(1 - ((X - x_optic_disc) / a_optic_disc[1]) ** 2)
    )
    d_interp[idx_optic_disc] = 0
    d_rho_theta = d_interp.T
    Rho = Rho.T
    Theta = Theta.T

    # == 2D Inverse Transform Sampling
    theta_vec = np.unique(Theta)
    rho_vec = np.unique(Rho)

    n_theta = theta_vec.size
    n_rho = rho_vec.size

    d_rho_theta_xRho = d_rho_theta * Rho
    d_theta = np.zeros(d_rho_theta.shape)
    for i in range(n_theta):
        d_theta[1:, i] = cumtrapz(d_rho_theta_xRho[:, i], rho_vec)

    Z = np.zeros(d_rho_theta.shape)
    for i in range(n_rho):
        Z[i, 1:] = cumtrapz(d_theta[i, :], theta_vec)
    scaling_factor = n_fibers_theor / Z[-1, -1]

    d_rho_theta_scaled = d_rho_theta * scaling_factor

    # computations are re-performed with the given scaling factor because we are in polar coordinates
    d_rho_theta_scaled_xRho = d_rho_theta_scaled * Rho
    d_theta_scaled = np.zeros(d_rho_theta.shape)
    for i in range(n_theta):
        d_theta_scaled[1:, i] = cumtrapz(d_rho_theta_scaled_xRho[:, i], rho_vec)

    Z_scaled = np.zeros(d_rho_theta.shape)
    for i in range(n_rho):
        Z_scaled[i, 1:] = cumtrapz(d_theta_scaled[i, :], theta_vec)
    normalization = Z_scaled[-1, -1]

    F_theta = Z_scaled[-1, :] / normalization

    d_theta = np.expand_dims(d_theta_scaled[-1, :], axis=0)
    d_theta_tiled = np.tile(d_theta, (n_rho, 1))
    F_rho_given_theta = d_theta_scaled / d_theta_tiled

    Finv_theta = interp1d(F_theta, theta_vec)

    u_theta = np.random.rand(n_fibers_theor, 1)
    u_rho = np.random.rand(n_fibers_theor, 1)

    y_theta = Finv_theta(u_theta)

    y_rho = np.zeros((n_fibers_theor, 1))
    for i in range(n_fibers_theor):
        idx_sampled_theta = np.where(F_theta >= u_theta[i])
        ind_sampled_theta = idx_sampled_theta[0][0]
        Finv_rho_given_theta = interp1d(F_rho_given_theta[:, ind_sampled_theta], rho_vec)
        y_rho[i] = Finv_rho_given_theta(u_rho[i])

    locs_rgc_ret_rho = y_rho
    locs_rgc_ret_theta = y_theta

    locs_rgc_ret_x = y_rho * np.cos(y_theta)
    locs_rgc_ret_y = y_rho * np.sin(y_theta)
    locs_rgc_ret = np.concatenate((locs_rgc_ret_x, locs_rgc_ret_y), axis=1)

    locs_rgc_opt_rho = np.sqrt(u_rho) * radius_optic_nerve
    locs_rgc_opt_theta = u_theta * 2 * np.pi

    locs_rgc_opt_x = locs_rgc_opt_rho * np.cos(locs_rgc_opt_theta)
    locs_rgc_opt_y = locs_rgc_opt_rho * np.sin(locs_rgc_opt_theta)
    locs_rgc_opt = np.hstack((locs_rgc_opt_x, locs_rgc_opt_y))

    # == Displace RGC soma locations to obtain receptive field center locations
    theta_array = np.linspace(0, 2 * np.pi, n_interp_theta)
    rho_array = np.linspace(0, radius_retina, n_interp_rho)

    locs_rgc_rf_theta = locs_rgc_ret_theta
    locs_rgc_rf_rho = locs_rgc_ret_rho

    sector_dtheta = np.max([np.pi / 500, theta_array[1]])

    sector_theta0 = np.arange(0, 2 * np.pi, sector_dtheta)
    sector_theta0 = sector_theta0  # avoid taking 0 and 2 * np.pi

    n_sectors = sector_theta0.size

    displacement_zone_limit = 4.034  # [mm], rho
    is_rgc_in_displacement_zone = locs_rgc_ret_rho <= displacement_zone_limit

    E2v = np.zeros(n_sectors)
    count_rf = np.zeros((n_interp_rho, n_sectors))
    for i in tqdm(range(n_sectors)):
        ind_theta = np.argmin(np.abs(theta_array - sector_theta0[i]))  # find index of theta closest to sector center
        ind_rho = np.argmin(np.abs(rho_array - 4.034))  # find index of rho closest to displacement zone limit

        # find index of highest rho with d_interp > 0 at given theta
        idx = np.where(d_interp[ind_theta, np.arange(ind_rho, 1, -1)] > 0)
        idx = idx[0][0]
        ind_rho_corrected = ind_rho - idx

        if i == 0:
            # indices of thetas between -dtheta/2 and dtheta/2
            idx_neg = np.logical_and(theta_array > 2 * np.pi - sector_dtheta / 2,
                                     theta_array < 2 * np.pi)  # index of thetas between 2*pi - dtheta/2 and 2*pi
            idx_pos = theta_array < sector_dtheta / 2  # index of thetas between 0 and dtheta/2
            is_theta_in_cone = np.logical_or(idx_neg, idx_pos)

            Theta_neg = Theta[:, idx_neg] - 2 * np.pi  # thetas between -dtheta/2 and 2*pi
            Theta_pos = Theta[:, idx_pos]
            Theta_in_sector = np.hstack((Theta_neg, Theta_pos))
            is_rgc_in_cone = np.logical_or(locs_rgc_ret_theta > 2 * np.pi - sector_dtheta / 2,
                                           locs_rgc_ret_theta < sector_dtheta / 2)
            idx_rgc_in_sector = np.logical_and(is_rgc_in_displacement_zone, is_rgc_in_cone)
        else:
            is_theta_in_cone = np.abs(theta_array - sector_theta0[i]) <= sector_dtheta / 2
            Theta_in_sector = Theta[:, is_theta_in_cone]
            is_rgc_in_cone = np.abs(locs_rgc_ret_theta - sector_theta0[i]) <= sector_dtheta / 2
            idx_rgc_in_sector = np.logical_and(is_rgc_in_displacement_zone, is_rgc_in_cone)

        Rho_in_sector = Rho[:, is_theta_in_cone]
        d_in_sector = d_rho_theta[:, is_theta_in_cone]
        n_theta_in_cone = np.sum(is_theta_in_cone)
        count_rgc = np.zeros((n_interp_rho, n_theta_in_cone))
        if n_theta_in_cone == 1:
            for j in range(n_theta_in_cone):
                vals = d_in_sector * Rho_in_sector * sector_dtheta
                count_rgc[1:, 0] = cumtrapz(vals[:, 0], rho_array)
        else:
            for j in range(n_theta_in_cone):
                count_rgc[1:, j] = cumtrapz(d_in_sector[:, j] * Rho_in_sector[:, j], rho_array)

            for k in range(n_interp_rho):
                count_rgc[k, 1:] = cumtrapz(count_rgc[k, :], Theta_in_sector[0, :])

        E2v[i] = brentq(f=compute_count_difference, a=0.1, b=3,
                        args=(rho_array, count_rgc, ind_rho_corrected, sector_dtheta))
        count_rf_temp = compute_count_rf(E2v[i], rho_array, sector_dtheta)
        count_rf[1:, i] = count_rf_temp

        cs = CubicSpline(rho_array, count_rgc[:, -1])
        C_rgc = cs(locs_rgc_ret_rho[idx_rgc_in_sector])

        cs = CubicSpline(count_rf[:, i], rho_array)
        locs_rgc_rf_rho[idx_rgc_in_sector] = cs(C_rgc)

    locs_rgc_rf_x, locs_rgc_rf_y = pol2cart(locs_rgc_rf_rho, locs_rgc_rf_theta)
    locs_rgc_rf = np.hstack((locs_rgc_rf_x, locs_rgc_rf_y))
    return locs_rgc_ret, locs_rgc_opt, locs_rgc_rf


def compute_rf_matrix(
        locs_rgc_rf, 
        scene_width_px=200, 
        scene_height_px=200, 
        radius_retina=10, 
        radius_retina_sphere=11.459
    ):
    """Compute the receptive field matrix."""
    scene_width_mm = 2 * radius_retina * np.power(1 + (scene_height_px / scene_width_px) ** 2, -1 / 2)
    scene_height_mm = scene_width_mm / scene_width_px * scene_height_px

    pixel_width_mm = scene_width_mm / scene_width_px
    pixel_height_mm = scene_height_mm / scene_height_px

    pixel_locs_x = np.arange(scene_width_px) * pixel_width_mm - scene_width_mm / 2 + pixel_width_mm / 2
    pixel_locs_y = np.arange(scene_height_px) * pixel_height_mm - scene_height_mm / 2 + pixel_height_mm / 2

    # x_pixels, y_pixels --> pixel_locs
    deg2mm = (2 * np.pi * radius_retina_sphere) / 360
    mm2deg = 1 / deg2mm

    [X, Y] = np.meshgrid(pixel_locs_x, pixel_locs_y)

    # Python flattens differently
    X = np.expand_dims(X.flatten(), axis=1)
    Y = np.expand_dims(Y.flatten(), axis=1)
    locs = np.hstack((X, Y))
    dx = pixel_locs_x[1] - pixel_locs_x[0]
    dy = pixel_locs_y[1] - pixel_locs_y[0]

    def kernel_sigma(E):
        return 10 * 0.0263 * (1 + 3.15 / (7.99 / (1 + 0.29 * E + 0.000012 * E ** 2))) / (
                    2 * np.sqrt(2 * np.log(2)))  # E in degrees

    def kernel_func(locs, locs_rgc_rf):
        locs_rgc_rf_rho = np.sqrt(np.sum(locs_rgc_rf ** 2, axis=0))
        sigma = kernel_sigma(locs_rgc_rf_rho * mm2deg) * deg2mm
        cov = sigma ** 2
        kernel = 300 * dx * dy * multivariate_normal.pdf(locs, mean=locs_rgc_rf, cov=cov)
        return kernel

    margin = 0.5
    idx_fibers_in_scene = np.where(
        np.logical_and(
            np.abs(locs_rgc_rf[:, 0]) <= scene_width_mm / 2 + margin,
            np.abs(locs_rgc_rf[:, 1]) <= scene_height_mm / 2 + margin
        )
    )
    idx_fibers_in_scene = idx_fibers_in_scene[0]

    col_idx = []
    row_idx = []
    vals = []
    n_support = []
    n_fibers = locs_rgc_rf.shape[0]
    
    for i in tqdm(range(n_fibers)):
        receptive_field = kernel_func(locs, locs_rgc_rf[i, :])
        rf_support = np.where(receptive_field > 1e-3)
        rf_support = rf_support[0]
        n_support.append(rf_support.size)
        col_idx.extend(rf_support)
        row_idx.extend(i * np.ones(n_support[i]))
        vals.extend(receptive_field[rf_support])

    rf_matrix = csc_matrix((vals, (row_idx, col_idx)), shape=(n_fibers, scene_width_px * scene_height_px))

    return rf_matrix


def compute_firing_rates_from_image(
        image, rf_matrix, n_ranges=255
    ):
    firing_rates = rf_matrix.dot(image.flatten() - 1) / (n_ranges - 1)
    return firing_rates


def compute_image_from_firing_rates(
        firing_rates, rf_matrix, h_im, w_im, n_ranges=255, 
    ):
    denominator = rf_matrix.sum(0)
    denominator[denominator == 0] = 1

    numerator = np.transpose(firing_rates) * rf_matrix
    image = np.floor(
        np.reshape(numerator / denominator, (h_im, w_im)) / 300 * (n_ranges - 1)
    ) + 1
    return image


def save_optic_geom_variables(
        r_ret,
        r_opt,
        radius_retina_sphere,
        locs_rgc_ret, 
        locs_rgc_opt, 
        locs_rgc_rf,
        scene_width_px,
        scene_height_px,
        rf_matrix,
        filename: str='optic_geom_variables.pkl'
    ):
    data = {}
    data["r_ret"] = r_ret
    data["r_opt"] = r_opt
    data["radius_retina_sphere"] = radius_retina_sphere
    data["locs_rgc_ret"] = locs_rgc_ret
    data["locs_rgc_opt"] = locs_rgc_opt
    data["locs_rgc_rf"] = locs_rgc_rf
    data["scene_width_px"] = scene_width_px
    data["scene_height_px"] = scene_height_px
    data["rf_matrix"] = rf_matrix
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_optic_geom_variables(
        filename: str='optic_geom_variables.pkl'
    ):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    r_ret = data["r_ret"]
    r_opt = data["r_opt"]
    radius_retina_sphere = data["radius_retina_sphere"]
    locs_rgc_ret = data["locs_rgc_ret"]
    locs_rgc_opt = data["locs_rgc_opt"]
    locs_rgc_rf = data["locs_rgc_rf"]
    scene_width_px = data["scene_width_px"]
    scene_height_px = data["scene_height_px"]
    rf_matrix = data["rf_matrix"]
    
    return (
        r_ret,
        r_opt,
        radius_retina_sphere,
        locs_rgc_ret, 
        locs_rgc_opt, 
        locs_rgc_rf,
        scene_width_px,
        scene_height_px,
        rf_matrix,
    )


def plot_rgcs(locs_rgc, FR=[None], ax=None, LBHFRR=265, n_ranges=12,
            FR_col_ranges=[None], alpha=None, marker_size=None, col=None):
    """Plots the RGCs as points.

    If `FR` is not specified, all RGCs are plotted with the same color.
    If `FR` is specified, RGCs are colored according to their
    corresponding firing rate in `FR`. Higher firing rate correspond
    to lighter green.
    RGCs are divided in groups according to their firing rate. RGCs
    that belong to the same group (so RGCs with a similar firing
    rate) are plotted with the same color. The number of groups is
    `n_ranges` and the firing rate ranges are computed so that the
    boundaries among firing rate levels are linearly spanned between 0
    and `LBHFRR`. All the firing rates greater that `LBHFRR` belong
    to the highest firing rate range. It is assumed that negative firing
    rates belong to the lowest firing rate range.
    Alternatively, it is possible to directly specify the boundaries
    for the firing rate ranges through the argument `FR_col_ranges`.

    Parameters
    ----------
    locs_rgc : (N,3) ndarray
        Locations of RGCs in any space (retina, optic nerve or
        receptive fields). Each row contains the location (x,y,z) of a
        RGC in cartesian coordinates (in mm).
    FR : (N) ndarray, optional
        Contains the firing rate of the RGC in the same order as in
        `locs_rgc`.
    ax : AxesSubplot, optional
        Axis handler. It specifies the destination axis of the plot.
    LBHFRR : int, default=265
        Lower boundary for the highest firing rate range.
    n_ranges : int, default=12
        Number of firing rate ranges for the plot.
    FR_col_ranges : (N) ndarray, optional
        Represents the values of the firing rate that separate the
        ranges.
        If not explicitly specified, `FR_col_ranges` is computed from
        `LBHFRR` and `n_ranges`, so that there are `n_ranges` firing
        rate ranges.
    """
    N_fibers = locs_rgc.shape[0]

    if marker_size is None:
        marker_size = plt.rcParams['lines.markersize'] ** 2  # size of the markers in the scatter plot

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    if alpha is None:
        alpha = .1

    # if `FR` is specified, then each fiber is plotted according to the
    # firing rate associated with it
    if len(FR) == N_fibers:
        # if no `FR_col_ranges` is specified
        if len(FR_col_ranges) == 1 and FR_col_ranges[0] is None:
            FR_col_ranges = np.linspace(0, LBHFRR, n_ranges)
            FR_col_ranges[0] = -np.inf
            FR_col_ranges = np.append(FR_col_ranges, np.inf)
        # update `n_ranges` and `LBHFRR` (needed if FR_col_ranges is
        # passed as argument)
        LBHFRR = FR_col_ranges[-2]
        n_ranges = len(FR_col_ranges) - 1
        # plot RGCs
        for i in range(n_ranges):
            idx = [j for j in range(N_fibers) if FR[j] > FR_col_ranges[i] and FR[j] <= FR_col_ranges[i + 1]]
            ax.scatter(
                locs_rgc[idx, 0], locs_rgc[idx, 1], facecolors=[0, max(FR_col_ranges[i], 0) / LBHFRR, 0],
                edgecolors=[0, max(FR_col_ranges[i], 0) / LBHFRR, 0],
                s=marker_size, alpha=alpha)
    # if `FR` is not specified, then plot every fiber with the same
    # color
    else:
        ax.scatter(locs_rgc[:, 0], locs_rgc[:, 1], s=marker_size, alpha=alpha, edgecolors=col, facecolors=col)

    ax.set_aspect('equal')
