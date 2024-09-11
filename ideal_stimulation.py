import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from math import floor

import optic_geom as og


def gen_gridcells_opt(L, grid_side=5.):
    """Generate the upper right and bottom left vertices of an
    inhomogeneous square grid.

    Parameters
    ----------
    L : list[int]
        Array of "normalized" cell sides (one for each ring), from center
        out.
    grid_side : float, default=5
        Side of the grid (in mm).

    Returns
    -------
    cell_corners : ndarray
        2-D ndarray of cell corner coordinates. First two columns are the
        x coordinates of the upper right and bottom left corners, while
        the last two columns represents their y coordinates.

    Notes
    -----
    The algorithm has some limitations, for example:
    * the grid is obtained by putting together one subgrid per
    quadrant, so in the inner ring there are at least 4 squares
    (cannot have the single center square)
    * only grids with L[i] | L[i+1] for all i are admissible (not a
    super limitation...)

    Examples
    --------
    >>> gen_gridcells_opt([6, 2, 2, 1, 1])
    array([[ 0.        ,  1.25      ,  0.        ,  1.25      ],
       [-1.25      ,  0.        ,  0.        ,  1.25      ],
       [-1.25      ,  0.        , -1.25      ,  0.        ],
       [ 0.        ,  1.25      , -1.25      ,  0.        ],
       [ 1.25      ,  1.66666667,  0.        ,  0.41666667],
       [ 1.25      ,  1.66666667,  0.41666667,  0.83333333],
       [ 1.25      ,  1.66666667,  0.83333333,  1.25      ],
       [ 0.        ,  0.41666667,  1.25      ,  1.66666667],
       [ 0.41666667,  0.83333333,  1.25      ,  1.66666667],
       [ 0.83333333,  1.25      ,  1.25      ,  1.66666667],
       ...
    """
    n_rings = len(L)

    # check that the grid is admissible
    for i in range(n_rings - 1):
        if L[i] % L[i + 1]:
            raise ValueError('Inadmissible grid!')

    UR = np.insert(np.cumsum(L), 0, 0)  # upper-right vertex of each ring
    ur = np.zeros((0, 2))  # upper-right cell vertices
    bl = np.zeros((0, 2))  # bottom-left cell vertices
    for i in range(n_rings):
        n_side = int(UR[i] / L[i])  # number of cells to be added on the side

        x_ur = np.tile(UR[i + 1], (1, n_side))
        y_ur = np.array([range(1, n_side + 1)]) * L[i]

        ur_new_v = np.hstack((x_ur.transpose(),
                              y_ur.transpose()))  # new vertical cells
        ur_new_h = np.hstack((y_ur.transpose(),
                              x_ur.transpose()))  # new horizontal cells
        ur_new_c = np.array([[UR[i + 1], UR[i + 1]]])  # new corner cell

        # build upper right vertices using reflection + translation
        ur_1_new = np.vstack((ur_new_v, ur_new_h, ur_new_c))
        ur_2_new = np.vstack((np.array([-ur_1_new[:, 0] + L[i]]),
                              np.array([ur_1_new[:, 1]]))).transpose()
        ur_3_new = np.vstack((np.array([-ur_1_new[:, 0] + L[i]]),
                              np.array([-ur_1_new[:, 1] + L[i]]))).transpose()
        ur_4_new = np.vstack((np.array([ur_1_new[:, 0]]),
                              np.array([-ur_1_new[:, 1] + L[i]]))).transpose()
        ur_new = np.vstack((ur_1_new, ur_2_new, ur_3_new, ur_4_new))
        ur = np.vstack((ur, ur_new))

        # bottom left from upper right is two translations
        bl_new = np.vstack((np.array([ur_new[:, 0] - L[i]]),
                            np.array([ur_new[:, 1] - L[i]]))).transpose()
        bl = np.vstack((bl, bl_new))

    return np.vstack((
        bl[:, 0],
        ur[:, 0],
        bl[:, 1],
        ur[:, 1],
    )).transpose() * grid_side / (2 * UR[-1])


def gen_gridcells(space='ret', **options):
    """Creates the four boundaries for each cell of the partition.

    Parameters
    ----------
    space : {'ret', 'opt'}, default='ret'
        It specifies in which space the partition takes place. If the
        space is `ret`, the configuration of the cells assumes a
        rectangular shape. If the space is `opt`, the partition is
        computed used the method `generate_gridcell`.
    **options : dict
        If space is `ret`, then the items `n_cells_w`, `n_cells_h`,
        `w_cell`, and `h_cell`, are used to specify the number of cells
        in the horizontal and vertical directions, the width and the
        height of the cells, respectively. The other items are ignored.
        If space is `opt`, then the item `L` is used to construct the
        grid using the method `gen_gridcells_opt`. The other items are
        ignored.
        # For both cases, if specified, the item `probability_swap` 
        # establishes the probability for 2 points to be swapped.
        # The pdf is intended to describe the probability
        # of swapping depending on the distance between two cells, using their
        # center as end points.

    Returns
    -------
    cell_corners : ndarray
      2-D ndarray of cell corner coordinates. First two columns are the
      x coordinates of the upper right and bottom left corners, while
      the last two columns represents their y coordinates.
    """
    # default key-value pairs
    options.setdefault('n_cells_w', 10)
    options.setdefault('n_cells_h', 6)
    options.setdefault('w_cell', .525)
    options.setdefault('h_cell', .525)
    options.setdefault('L', [6, 2, 2, 1, 1])
    options.setdefault('grid_side', 5.)

    # partition using the partitioning grid of the optic nerve
    if space == 'opt':
        L = options['L']
        grid_side = options['grid_side']
        cell_corners = gen_gridcells_opt(L, grid_side)
    # partition using the partitioning grid of the retinal implant
    else:
        n_cells_w = options['n_cells_w']
        n_cells_h = options['n_cells_h']
        w_cell = options['w_cell']
        h_cell = options['h_cell']

        # width of the partitioning square (width of the implant)
        L_w = w_cell * (n_cells_w + 1)
        # height of the partitioning square (height of the implant)
        L_h = w_cell * (n_cells_h + 1)

        x_cells = np.transpose((np.array(range(0, n_cells_w)) * w_cell
                                - L_w / 2 + w_cell / 2))
        # inverted y axis for the image convention
        y_cells = np.transpose((np.array(range(n_cells_h - 1, -1, -1)) * h_cell
                                - L_h / 2 + h_cell / 2))

        left_boundary = x_cells - w_cell / 2
        bottom_boundary = y_cells - h_cell / 2
        right_boundary = x_cells + w_cell / 2
        upper_boundary = y_cells + h_cell / 2

        x_lower, y_lower = np.meshgrid(left_boundary, bottom_boundary)
        x_upper, y_upper = np.meshgrid(right_boundary, upper_boundary)

        cell_corners = np.vstack((
            x_lower.flatten(),
            x_upper.flatten(),
            y_lower.flatten(),
            y_upper.flatten()
        )).transpose()

    # reorder cell_corners
    sorted_indices = np.lexsort((cell_corners[:, 1], -cell_corners[:, 3]))
    cell_corners = cell_corners[sorted_indices]

    return cell_corners


def gen_cells(vision_type, L, locs_rgc_opt, locs_rgc_ret):
    """Generates cells and gets all the indexes of the RGCs
    inside the cells.
    """
    # creation cells for stimulation
    if vision_type == 'optic nerve':
        cell_corners = gen_gridcells(space='opt', L=L)
        locs_rgc = locs_rgc_opt
    elif vision_type == 'retina':
        cell_corners = gen_gridcells('ret')
        locs_rgc = locs_rgc_ret
    n_cells = cell_corners.shape[0]

    if vision_type == 'optic nerve' or vision_type == 'retina':
        idxs_inside_cells = compute_idxs_inside_cells(
            cell_corners=cell_corners,
            locs_rgc=locs_rgc
        )

    cell_centers = np.hstack((
        np.mean(cell_corners[:, 0:2], axis=1)[:, None],
        np.mean(cell_corners[:, 2:], axis=1)[:, None]
    ))

    return cell_corners, cell_centers, idxs_inside_cells, n_cells, locs_rgc


def get_target_image(
        image_original, 
        im_width=20 * np.sqrt(2), 
        image_original_ratio=1, 
        n_ranges=12,
        w_im=200, 
        h_im=200, 
        w_shift=0, 
        h_shift=0
    ):
    """Padding of the original image (`image_original`) and remap of its
    values in the interval 1-`n_ranges`.

    Parameters
    ----------
    image_original : ~PIL.Image.Image
        Original image.
    im_width : float, default=20 * sqrt(2)
        Width of the output image in the space of the receptive fields
        (in mm).
    image_original_ratio : float, default=1
        Ratio between the desired width of the original image and that
        of the output image in the space of the receptive fields.
    n_ranges : int, default=12
        Number of gray levels of the image.
    w_im : int, default=200
        Width of the output image (in pixels).
    h_im : int, default=200
        Height of the output image (in pixels).
    w_shift : float, default=0
        Horizontal shift of the original image in the space of the
        receptive fields (mm).
        Positive values make the original image shift leftwards.
    h_shift : float, default=0
        Vertical shift of the original image in the space of the
        receptive fields (mm).
        Positive values make the original image shift upwards.

    Returns
    -------
    image : (h_im, w_im) ndarray
        Target image.
    """
    # resize image
    w_image_original, h_image_original = image_original.size
    # width in pixels of the scaled image
    w_ims = round(image_original_ratio * w_im)
    # height in pixels of the scaled image
    h_ims = round(w_ims * h_image_original / w_image_original)

    im_gray = ImageOps.grayscale(image_original)
    im_scaled = im_gray.resize((w_ims, h_ims))

    # convert image to numpy array with values between 1 and `n_ranges` for
    # each pixel
    im_scaled = np.round(np.asarray(im_scaled) / 255 * (n_ranges - 1) + 1)

    # Image padding
    # The output image must be a h_im-by-w_im array, thus, in general,
    # padding is needed.
    # After its initialization, the vector `image` is properly overwritten
    # with the scaled image (`im_scaled`). In general, the borders of
    # the scaled image may go beyond those of the image (`image`).
    # In the following, the name of the indexes referring to `image` start
    # with "idx", while those referring to `image` are called `idx_s`.
    # The suffixes "_w" and "_h" refer to the width and the height of
    # the image, thus they correspond to axes 1 and 0 if the arrays,
    # respectively.
    side_pix = im_width / w_im  # side length of a single pixel (mm)
    idx_s_w_start = 0
    idx_s_w_stop = im_scaled.shape[1]
    idx_s_h_start = 0
    idx_s_h_stop = im_scaled.shape[0]
    idx_w_start = floor((w_im - w_ims) / 2 - w_shift / side_pix)
    idx_w_stop = floor((w_im + w_ims) / 2 - w_shift / side_pix)
    idx_h_start = floor((h_im - h_ims) / 2 - h_shift / side_pix)
    idx_h_stop = floor((h_im + h_ims) / 2 - h_shift / side_pix)

    # Handling cases in which the scaled image goes out of the
    # image's boundaries.
    if idx_w_start < 0:
        idx_s_w_start = -idx_w_start
        idx_w_start = 0
    if idx_w_stop > w_im:
        idx_s_w_stop = idx_s_w_stop - (idx_w_stop - w_im)
        idx_w_stop = w_im
    if idx_h_start < 0:
        idx_s_h_start = -idx_h_start
        idx_h_start = 0
    if idx_h_stop > h_im:
        idx_s_h_stop = idx_s_h_stop - (idx_h_stop - h_im)
        idx_h_stop = h_im

    image = np.ones((h_im, w_im))  # padding with 1s (min gray level is 1)
    image[idx_h_start:idx_h_stop, idx_w_start:idx_w_stop] = im_scaled[idx_s_h_start: idx_s_h_stop,
                              idx_s_w_start: idx_s_w_stop]

    return image


def compute_idxs_inside_cells(cell_corners, locs_rgc, return_idxs_outside_cells=False):
    """Finds the idxs of the RGCs that are inside each cell.    
    """
    n_fibers = locs_rgc.shape[0]
    n_cells = len(cell_corners)
    idxs_inside_cells = [None for _ in range(n_cells)]
    idxs_inside_any_cell = []
    for i in range(n_cells):
        idxs_inside_cells[i] = [
            j for j in range(n_fibers) if (
                cell_corners[i, 0] <= locs_rgc[j, 0] < cell_corners[i, 1] and
                cell_corners[i, 2] <= locs_rgc[j, 1] < cell_corners[i, 3]
            )
        ]
        idxs_inside_any_cell.extend(idxs_inside_cells[i])
    if return_idxs_outside_cells:
        idxs_outside_cells = [i for i in range(n_fibers) if i not in idxs_inside_any_cell]
        return idxs_inside_cells, idxs_outside_cells
    return idxs_inside_cells


def ideal_fr_generator(
        image, locs_rgc, cell_corners, rf_matrix, idxs_inside_cells, 
        n_ranges=12
    ):
    """Generates the firing rates associated with an ideal stimulation.

    The input image `image` is used to compute the firing rate of the RGCs
    according to their receptive fields contained in the `rf_matrix`
    matrix. `fr_natural` collects the obtained firing rates.
    To simulate the ideal stimulation, all RGCs inside a cell assume
    the same firing rate computed as the median of the firing rates
    of the correspondent values of `fr_natural`. The firing rate values
    computed in this way are collected in `fr_ideal_stim`.
    The perceived image `perception` is computed from the receptive
    fields using `fr_ideal_stim` for the firing rates.
    Boundaries for cells are specified in `cell_corners` and must
    respect conventions of the module `gen_gridcells` output.

    Parameters
    ----------
    image : (N,2) ndarray
        Target image.
    locs_rgc : (N,3) ndarray
        Locations of RGCs in any space (retina, optic nerve or
        receptive fields). Each row contains the location (x,y,z) of a
        RGC in cartesian coordinates (in mm).
    cell_corners : ndarray
        2-D ndarray of cell corner coordinates. First two columns are the
        x coordinates of the upper right and bottom left corners, while
        the last two columns represents their y coordinates.
    rf_matrix : csc_matrix
        N-by-P sparse matrix representing the receptive fields
        of the RGCs, where N is the number of fibers and P is the
        number of pixels. Each row contains the vectorization of the
        receptive field of one RGC.
    idxs_inside_cells : list[list[int]] 
        List of the indexes of the RGCs grouped by cells. 
        Each element of `idxs_inside_cells` contains the list of the 
        indexes of RGCs inside the corresponding cell.
    n_ranges : int, default=12
        Number of firing rate ranges for the plot.

    Returns
    -------
    perception : ndarray
        Image that is generated from the partition. It has the same
        shape as `image`.
    fr_natural : (N) ndarray
        Array that contains the firing rate associated to the
        perception of the target image.
    fr_ideal_stim : (N) ndarray
        Array that contains the firing rate associated to the ideal
        stimulation.
    """
    n_fibers = locs_rgc.shape[0]
    h_im, w_im = image.shape

    # compute FRs target
    fr_natural = rf_matrix.dot(image.T.flatten() - 1) / (n_ranges - 1)

    # create "optic nerve image" and re-compute FRs with the "optic
    # image"
    n_cells = len(cell_corners)
    fr_median = np.zeros((n_cells))  # median FR of RGCs in the cells
    fr_ideal_stim = np.zeros((n_fibers, 1))  # initialization

    for i in range(n_cells):
        # if there are no RGCs in the cell, put a 0 as median firing rate
        if len(idxs_inside_cells[i]) == 0:
            fr_median[i] = 0
        else:
            fr_median[i] = np.median(fr_natural[idxs_inside_cells[i]])

        fr_ideal_stim[idxs_inside_cells[i]] = fr_median[i]

    denominator = rf_matrix.sum(0)
    denominator[denominator == 0] = 1

    # create the perceived image after the discretization in the optic
    # nerve
    numerator = np.transpose(fr_ideal_stim) * rf_matrix
    perception = np.floor(
        np.reshape(numerator / denominator, (h_im, w_im)) / 300 * (n_ranges - 1)
    ).T + 1

    return perception, fr_natural, fr_ideal_stim


def compute_discretized_image(
        image, rf_matrix, idxs_inside_cells, scene_height_px, scene_width_px, n_ranges=12,
        idxs_inside_cells_after_global_shuffle=None,
        some_off=False, n_off=None, idxs_cells_off=None,
        return_n=False, idxs_dead_rgc=None,
    ):
    """Compute the perception that is generated from the model.
    
    The perception of a target image `image` is computed from the
    geometrical model and the grid.

    Parameters
    ----------
    image :
        Original image.
    rf_matrix : 
        N-by-P sparse matrix representing the receptive fields
        of the RGCs, where N is the number of fibers and P is the
        number of pixels. Each row contains the vectorization of the
        receptive field of one RGC.
    idxs_inside_cells : list[list[int]] 
        List of the indexes of the RGCs grouped by cells. 
        Each element of `idxs_inside_cells` contains the list of the 
        indexes of RGCs inside the corresponding cell.
    scene_height_px : 
        Height of the image to be generated. Must be the same used to
        create `rf_matrix`.
    scene_width_px : 
        Width of the image to be generated. Must be the same used to
        create `rf_matrix`.
    n_ranges : int, default=12
        Number of firing rate ranges for the plot.
    idxs_inside_cells_after_global_shuffle : list[list[int]] 
        List of the indexes of the RGCs grouped by cells after global
        shuffling. 
        Each element of `idxs_inside_cells_after_global_shuffle` 
        contains the list of the indexes of the RGCs inside the 
        corresponding cell, after the global shuffling is applied.
    some_off : 
        If true, some the firing rate of the RGCs inside some cells 
        is forced to be 0, independently on the target image. This 
        simulates damaged cells (sites).
    n_off : 
        Number of damaged cells to simulate.
    idxs_cells_off : list[int]
        List of the indexes of the damaged cells.
    return_n : 
        If True, `n_off` is returned, too.
    idxs_dead_rgc : list[int]
        List of the indexes of the RGCs that are assumed to be death.


    Return
    ------
    perception : 
        Perceived image.
    ideal_firing_rates : 
        Ideal firing rate.
    """
    natural_firing_rates = rf_matrix.dot(image.flatten() - 1) / (n_ranges - 1)

    n_fibers = natural_firing_rates.shape[0]
    n_cells = len(idxs_inside_cells)
    cellwise_firing_rates = np.zeros(n_cells)  # median FR of RGCs in the cells
    ideal_firing_rates = np.zeros(n_fibers)  # initialization

    for i in range(n_cells):
        # if there are no RGCs in the cell, use 0 as the cell firing rate
        if len(idxs_inside_cells[i]) == 0:
            cellwise_firing_rates[i] = 0
        else:
            firing_rates_in_cell = natural_firing_rates[idxs_inside_cells[i]]
            n_rgc_on = np.count_nonzero(firing_rates_in_cell)
            n_rgc_tot = firing_rates_in_cell.size
            on_proportion = n_rgc_on / n_rgc_tot
            if on_proportion > 0.3:
                cellwise_firing_rates[i] = np.max(firing_rates_in_cell)
            else:
                cellwise_firing_rates[i] = 0
    
    # if specified, simulate that some cells are not functioning
    cellwise_firing_rates, n_off = turn_some_cells_off(
        cellwise_firing_rates, 
        some_off=some_off, n_off=n_off, idxs_cells_off=idxs_cells_off
    )
    n = n_off

    # compute perception
    if idxs_inside_cells_after_global_shuffle is None:
        # if there is no shuffling, the idxs do not change
        idxs_inside_cells_after_global_shuffle = idxs_inside_cells
    for i in range(n_cells):
        ideal_firing_rates[idxs_inside_cells_after_global_shuffle[i]] = cellwise_firing_rates[i]
    
    if idxs_dead_rgc is not None:
        ideal_firing_rates[idxs_dead_rgc] = 0
        
    perception = og.compute_image_from_firing_rates(
        ideal_firing_rates, 
        rf_matrix, 
        scene_height_px, 
        scene_width_px, 
        n_ranges
    )
    
    if return_n:
        return perception, ideal_firing_rates, n
    return perception, ideal_firing_rates


def turn_some_cells_off(cellwise_firing_rates, 
             some_off=False, n_off=None, idxs_cells_off=None):
    some_off_cellwise_firing_rates = np.copy(cellwise_firing_rates)
    if some_off or n_off is not None or idxs_cells_off is not None:
        if idxs_cells_off is None:
            n_cells = len(cellwise_firing_rates)
            if n_off is None:
                n_off = int(n_cells / 10)
            idxs_cells_off = np.random.choice(n_cells, size=n_off, replace=False)
        else:
            if n_off is not None and n_off != len(idxs_cells_off):
                print("\nMismatch between n_off and the number of indexes input.\nIgnoring input n_off.\n")
        some_off_cellwise_firing_rates[idxs_cells_off] = 0
    return some_off_cellwise_firing_rates, n_off


def compute_discretized_firing_rates(image, rf_matrix, idxs_inside_cells, n_ranges=255):
    natural_firing_rates = rf_matrix.dot(image.flatten() - 1) / (n_ranges - 1)

    n_fibers = natural_firing_rates.shape[0]
    n_cells = len(idxs_inside_cells)
    cellwise_firing_rates = np.zeros(n_cells)  # median FR of RGCs in the cells
    ideal_firing_rates = np.zeros(n_fibers)  # initialization

    for i in range(n_cells):
        # if there are no RGCs in the cell, put a 0 as cell-wise firing rate
        if len(idxs_inside_cells[i]) == 0:
            cellwise_firing_rates[i] = 0
        else:
            firing_rates_in_cell = natural_firing_rates[idxs_inside_cells[i]]
            n_rgc_on = np.count_nonzero(firing_rates_in_cell)
            n_rgc_tot = firing_rates_in_cell.size
            on_proportion = n_rgc_on / n_rgc_tot
            if on_proportion > 0.3:
                cellwise_firing_rates[i] = np.max(firing_rates_in_cell)
            else:
                cellwise_firing_rates[i] = 0

        ideal_firing_rates[idxs_inside_cells[i]] = cellwise_firing_rates[i]
    return ideal_firing_rates


def plot_grid(cell_corners, colors=None, grid_color=None, idxs_cells_off=None, 
              color_cross=None, ax=None):
    if ax is None:
        ax = plt.gca()
        
    n_gridcells = cell_corners.shape[0]
    widths = cell_corners[:, 1] - cell_corners[:, 0]
    heights = cell_corners[:, 3] - cell_corners[:, 2]
    
    if colors is None:
        if grid_color is None:
            grid_color = 'red'
        for i in range(n_gridcells):
            rectangle = plt.Rectangle(
                (cell_corners[i, 0], cell_corners[i, 2]),
                widths[i], heights[i],
                fc=[1, 1, 1, 0],
                ec=grid_color,
                zorder=2
            )
            ax.add_patch(rectangle)
    else:
        if grid_color is None:
            grid_color = 'white'
        for i in range(n_gridcells):
            rectangle = plt.Rectangle(
                (cell_corners[i, 0], cell_corners[i, 2]),
                widths[i], heights[i],
                fc=[colors[i], colors[i], colors[i], 1],
                ec=grid_color,
                zorder=2
            )
            ax.add_patch(rectangle)

    if idxs_cells_off is not None:
        if color_cross is None:
            color_cross = 'red'
        for idx in idxs_cells_off:
            x1 = cell_corners[idx, 0]
            x2 = cell_corners[idx, 1]
            y1 = cell_corners[idx, 2]
            y2 = cell_corners[idx, 3]
            ax.plot([x1, x2], [y1, y2], color=color_cross, linewidth=2)
            ax.plot([x1, x2], [y2, y1], color=color_cross, linewidth=2)

    ax.axis("equal")


def load_target_image(
        filename,
        scene_width_px,
        scene_height_px,
        w_shift=0,
        h_shift=0,
        n_ranges=255
    ):
    """Load image from a '.npy' file."""

    image = Image.fromarray(np.load(filename))

    return get_target_image(
        image_original=image,
        image_original_ratio=0.75,
        w_im=scene_width_px,
        h_im=scene_height_px,
        w_shift=w_shift,
        h_shift=h_shift,
        n_ranges=n_ranges
    )


class TargetImagesLoadingClass:
    """Auxiliary class to load the target scenes."""
    
    n_scenes = 1
    target_images_folder = ".\\target_images\\"
    scene_width_px = 384
    scene_height_px = 216
    w_shift = 0
    h_shift = 0
    n_ranges = 255
    idx_images = np.arange(n_scenes)

    def __init__(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    setattr(self, k, v)
        if kwargs is not None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def get_all_settings(self):
        
        # Get all attributes of the class
        all_attributes = dir(self.__class__)

        # Get special attributes and methods to exclude
        exclude_names = [attr for attr in all_attributes if attr.startswith('__')]

        # Get values from instance and class dictionaries
        instance_values = vars(self)
        class_values = {key: value for key, value in self.__class__.__dict__.items() 
                        if not (callable(value) or key in exclude_names)}

        # Combine dictionaries and filter out unwanted names
        all_values = {key: value for key, value in {**instance_values, **class_values}.items() if key not in exclude_names}

        # Return values as a tuple
        return tuple(all_values.values())
    
    @classmethod
    def load_scenes(cls, *args, **kwargs):
        """Loads the target images."""

        load_object = cls(*args, **kwargs)
        
        # load data
        images = [None for _ in range(load_object.n_scenes)]
        for i, idx in enumerate(load_object.idx_images):
            image = np.load(load_object.target_images_folder + "stimulus" + str(idx) + ".npy")
            images[i] = Image.fromarray(image)

        # scenes creation
        scenes = [
            get_target_image(
                image_original=images[i_scene],
                image_original_ratio=0.75,
                w_im=load_object.scene_width_px,
                h_im=load_object.scene_height_px,
                w_shift=load_object.w_shift,
                h_shift=load_object.h_shift,
                n_ranges=load_object.n_ranges
            ) for i_scene in range(load_object.n_scenes)
        ]

        return scenes
