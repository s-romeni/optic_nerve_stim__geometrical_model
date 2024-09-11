import numpy as np
from scipy.sparse import csc_matrix
from scipy.spatial import distance_matrix

import optic_geom as og
import ideal_stimulation as ideal_stim

r_opt_default = 2.5


def generate_probability_swap_matrix(locations, idxs_to_swap_1=None, proximity_factor=None):
    """Generate the probability swap matrix using a gaussian kernel."""
    if idxs_to_swap_1 is None:
        idxs_to_swap_1 = np.arange(0, locations.shape[0])
    
    if locations.ndim == 1:
        locations = locations[:, None]

    dm = distance_matrix(locations[idxs_to_swap_1], locations)

    if proximity_factor is None:
        average_distance = np.mean(dm)
        proximity_factor = average_distance / 5

    weights = np.exp(-(dm ** 2) / (2 * proximity_factor ** 2))
    i_zeros, j_zeros = np.where(dm==0)
    weights[i_zeros, j_zeros] = 0

    p = weights / np.sum(weights, axis=1)[:, None]

    return p


def swap_array(
        array, 
        n_swaps=0, 
        probability_swap_matrix=None, 
        locations=None, 
        proximity_factor=None,
        return_idxs=False
    ):
    """Swap the elements of an array an arbitrary number of times.
    
    Swap the elements of `array`, `n_swaps` times, according to 
    `probability_swap_matrix`, which represent the probability that
    any pair of elements in `array` swap.
    If `probability_swap_matrix` is not specified, it is computed
    based on the distance between each element in `locations`,
    using a gaussian kernel with `proximity_factor` as sigma. In this
    case, if `locations` is not specified, `array` is used instead and
    if `proximity_factor` is not specified, a fifth of the average
    distance among the elements is used.

    Parameters
    ----------
    array : 
        Set of elements to be swapped.
    n_swaps : 
        Number of swap to perform
    probability_swap_matrix : 
        Square matrix representing the swapping probability between two 
        elements of `array`.
    locations : 
        Locations corresponding to the elements in `array`.
    proximity_factor : 
        Sigma for the gaussian kernel function determining the 
        probability-of-swapping matrix.
    return_idxs : 
        If True, the indexes used for the swaps are returned, too.
    """
    is_sparse = False
    if isinstance(array, csc_matrix):
        original_matrix = array
        array = list(range(original_matrix.shape[0]))
        is_sparse = True
    
    is_ndarray = False
    if isinstance(array, np.ndarray):
        array = array.tolist()
        is_ndarray = True
    
    output_array = [a for a in array]
    idxs_to_swap_1 = np.random.choice(range(len(array)), n_swaps, replace=False)

    if probability_swap_matrix is None and n_swaps > 0:
        # if no probability is specified for the swap, it is automatically
        # generated based on the distance
        if locations is None:
            # if locations is not specified, assume that the swapping must 
            # be applied to the array containing the distances
            locations = array  
        
        probability_swap_matrix = generate_probability_swap_matrix(
            locations, idxs_to_swap_1=idxs_to_swap_1, proximity_factor=proximity_factor
        )

    idxs_to_swap_2 = [None for _ in idxs_to_swap_1]
    for i, idx_to_swap_1 in enumerate(idxs_to_swap_1):
        idx_to_swap_2 = int(np.random.choice(range(len(array)), 1, replace=False, p=probability_swap_matrix[i])[0])
        tmp = output_array[idx_to_swap_1]
        output_array[idx_to_swap_1] = output_array[idx_to_swap_2]
        output_array[idx_to_swap_2] = tmp
        idxs_to_swap_2[i] = idx_to_swap_2
    
    if is_ndarray:
        if return_idxs:
            return np.array(output_array), idxs_to_swap_1, idxs_to_swap_2
        return np.array(output_array)
    if is_sparse:
        if return_idxs:
            return original_matrix[output_array, :], idxs_to_swap_1, idxs_to_swap_2
        return original_matrix[output_array, :]
    if return_idxs:
        return output_array, idxs_to_swap_1, idxs_to_swap_2
    return output_array


def local_shuffle(
        scene, rf_matrix, n_swaps_local_shuffle, locs_rgc,
        idxs_inside_cells, scene_height_px, scene_width_px, n_ranges=12, 
        return_idxs=True
    ):
    """Reconstruct a perception, given the image, the model, and the local
    shuffling parameters.
    
    The perception of a given target image `scene` is reconstructed, based on
    the geometrical model, the parameters of the ideal stimulation and the
    parameters of a local shuffle scenario.

    Parameters
    ----------
    scene : 
        Original image.
    rf_matrix : 
        N-by-P sparse matrix representing the receptive fields
        of the RGCs, where N is the number of fibers and P is the
        number of pixels. Each row contains the vectorization of the
        receptive field of one RGC.
    n_swaps_local_shuffle : 
        Number of swaps.
    locs_rgc : 
        Locations of RGCs in any domain. 
        Each row contains the location (x,y,z) of a RGC (in mm).
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
    return_idxs : 
        If True, the indexes used for the swaps are returned, too.
    
    Returns
    -------
    perception_local_shuffle : 
        Perception with local shuffle.
    rf_matrix_local_shuffle : 
        Receptive field matrix with local shuffle.
    idxs_to_swap_1 : 
        First indexes of the swapped RGCs' receptive field.
    idxs_to_swap_2 : 
        Second indexes of the swapped RGCs' receptive field.
    """
    rf_matrix_local_shuffle, idxs_to_swap_1, idxs_to_swap_2 = swap_array(
        rf_matrix, n_swaps=n_swaps_local_shuffle, probability_swap_matrix=None, 
        proximity_factor=None, locations=locs_rgc, return_idxs=return_idxs
    )
    perception_local_shuffle, _ = ideal_stim.compute_discretized_image(
        image=scene,
        rf_matrix=rf_matrix_local_shuffle,
        idxs_inside_cells=idxs_inside_cells,
        scene_height_px=scene_height_px,
        scene_width_px=scene_width_px,
        n_ranges=n_ranges,
    )
    
    if return_idxs:
        return perception_local_shuffle, rf_matrix_local_shuffle, idxs_to_swap_1, idxs_to_swap_2
    return perception_local_shuffle, rf_matrix_local_shuffle


def generate_global_shuffle(
        n_swaps_global_shuffle, locs_rgc,
        L_shuffle=None, grid_side_shuffle=None,
        proximity_factor=None
    ):
    """Generate the global shuffling.
    
    The domain is divided in a grid of cells and the N RGCs from one 
    cell are randomly swapped with the M RGCs of another cell, so that 
    their relative position with the cell center is maintained.
    This process is repeated `n_swaps_global_shuffle` times.
    """
    if L_shuffle is None:
        L_shuffle = [1, 1, 1]
    if grid_side_shuffle is None:
        grid_side_shuffle = 2 * r_opt_default / np.sqrt(2)

    cell_corners_global = ideal_stim.gen_gridcells(space='opt', L=L_shuffle, grid_side=grid_side_shuffle)
    cell_centers_global = np.hstack((
        np.mean(cell_corners_global[:, 0:2], axis=1)[:, None],
        np.mean(cell_corners_global[:, 2:], axis=1)[:, None]
    ))
    n_cells_global = len(cell_corners_global)
    cell_corners_global_shuffled = swap_array(
        cell_corners_global,
        n_swaps=n_swaps_global_shuffle,
        locations=cell_centers_global,
        proximity_factor=proximity_factor,
    )
    cell_centers_global_shuffled = np.hstack((
        np.mean(cell_corners_global_shuffled[:, 0:2], axis=1)[:, None],
        np.mean(cell_corners_global_shuffled[:, 2:], axis=1)[:, None]
    ))
    idxs_inside_cells_global, idxs_outside_cells_global = ideal_stim.compute_idxs_inside_cells(
        cell_corners=cell_corners_global,
        locs_rgc=locs_rgc,
        return_idxs_outside_cells=True
    )
    locs_rgc_global_shuffled = locs_rgc.copy()
    for idxs_inside_cell_global, cell_center_global, cell_center_global_shuffled in \
        zip(idxs_inside_cells_global, cell_centers_global, cell_centers_global_shuffled):
        locs_rgc_global_shuffled[idxs_inside_cell_global, :] = \
            locs_rgc[idxs_inside_cell_global, :] - cell_center_global + cell_center_global_shuffled
        

    return (
        cell_centers_global, n_cells_global,
        locs_rgc_global_shuffled, idxs_inside_cells_global, idxs_outside_cells_global,
        cell_centers_global_shuffled, cell_corners_global
    )


def generate_idxs_inside_cells_after_global_shuffle(
        n_swaps_global_shuffle, locs_rgc, cell_corners,
        L_shuffle=None, grid_side_shuffle=None,
        proximity_factor=None
    ):
    
    cell_centers_global, n_cells_global, \
        locs_rgc_global_shuffled, idxs_inside_cells_global, idxs_outside_cells_global, \
        cell_centers_global_shuffled, cell_corners_global = generate_global_shuffle(n_swaps_global_shuffle, locs_rgc,
                                                   L_shuffle=L_shuffle, grid_side_shuffle=grid_side_shuffle,
                                                   proximity_factor=proximity_factor)

    # compute new idxs_inside_cells
    idxs_inside_cells_after_global_shuffle = ideal_stim.compute_idxs_inside_cells(
        cell_corners=cell_corners,
        locs_rgc=locs_rgc_global_shuffled
    )

    return (
        idxs_inside_cells_after_global_shuffle, cell_centers_global, n_cells_global,
        locs_rgc_global_shuffled, idxs_inside_cells_global, idxs_outside_cells_global,
        cell_centers_global_shuffled, cell_corners_global
    )


def global_shuffle(scene, rf_matrix, n_swaps_global_shuffle, locs_rgc,
                  scene_height_px, 
                  scene_width_px, cell_corners,
                  L_shuffle=None, grid_side_shuffle=None, proximity_factor=None, n_ranges=12):
    """Reconstruct a perception, given the image, the model, and the global
    shuffling parameters.
    
    The perception of a given target image `scene` is reconstructed, based on
    the geometrical model, the parameters of the ideal stimulation and the
    parameters of a global shuffle scenario.

    Parameters
    ----------
    scene : 
        Original image.
    rf_matrix : 
        N-by-P sparse matrix representing the receptive fields
        of the RGCs, where N is the number of fibers and P is the
        number of pixels. Each row contains the vectorization of the
        receptive field of one RGC.
    n_swaps_global_shuffle : 
        Number of swaps.
    locs_rgc : 
        Locations of RGCs in any domain. 
        Each row contains the location (x,y,z) of a RGC (in mm).
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
        cell_corners
    L_shuffle : 
        Array of "normalized" cell sides (one for each ring), from center
        out, used for the grid of the global shuffle.
    grid_side_shuffle : 
        Side of the grid for the global shuffle.
    proximity_factor : 
        Sigma for the gaussian kernel function determining the 
        probability-of-swapping matrix.
    n_ranges : int, default=12
        Number of firing rate ranges for the plot.
    
    Returns
    -------
    perception_global_shuffle : 
        Perception with global shuffle.
    idxs_inside_cells_after_global_shuffle : list[list[int]] 
        List of the indexes of the RGCs grouped by cells after global
        shuffling. 
        Each element of `idxs_inside_cells_after_global_shuffle` 
        contains the list of the indexes of the RGCs inside the 
        corresponding cell, after the global shuffling is applied.
    cell_centers_global : 
        List of the centers of the cells used for global shuffling.
    n_cells_global : 
        Number of cells used for global shuffling.
    locs_rgc_global_shuffled : 
        Global shuffle of the location of the RGCs.
    idxs_inside_cells_global : list[list[int]] 
        List of the indexes of the RGCs grouped by the cells used for
        the global shuffling, after the global shuffling. 
        Each element of `idxs_inside_cells_global` contains the list 
        of the indexes of RGCs inside the corresponding cell of the 
        global shuffling grid, after the global shuffling.
    idxs_outside_cells_global : list[int]
        List of the RGCs that are outside the global shuffling grid.
    cell_centers_global_shuffled : 
        Centers of the cells of the global shuffling grid, after the 
        global shuffling.
    """
    idxs_inside_cells_after_global_shuffle, cell_centers_global, n_cells_global, \
        locs_rgc_global_shuffled, idxs_inside_cells_global, idxs_outside_cells_global, \
        cell_centers_global_shuffled, _ = \
        generate_idxs_inside_cells_after_global_shuffle(
            n_swaps_global_shuffle, locs_rgc, cell_corners,
            L_shuffle, grid_side_shuffle, proximity_factor
            )
    
    perception_global_shuffle, _ = ideal_stim.compute_discretized_image(
        image=scene,
        rf_matrix=rf_matrix,
        idxs_inside_cells=idxs_inside_cells_after_global_shuffle,
        scene_height_px=scene_height_px,
        scene_width_px=scene_width_px,
        n_ranges=n_ranges,
        )

    return (
        perception_global_shuffle, 
        idxs_inside_cells_after_global_shuffle, 
        cell_centers_global, 
        n_cells_global, 
        locs_rgc_global_shuffled, 
        idxs_inside_cells_global, 
        idxs_outside_cells_global, 
        cell_centers_global_shuffled
    )


def find_cells_inside_optic_nerve(cell_corners, radius_opt):
    x1 = cell_corners[:, 0]
    x2 = cell_corners[:, 1]
    y1 = cell_corners[:, 2]
    y2 = cell_corners[:, 3]
    _, rho_points1 = og.cart2pol(x1, y1)
    _, rho_points2 = og.cart2pol(x1, y2)
    _, rho_points3 = og.cart2pol(x2, y1)
    _, rho_points4 = og.cart2pol(x2, y2)
    return np.logical_and(
        np.logical_and(rho_points1 <= radius_opt, 
                       rho_points2 <= radius_opt),
        np.logical_and(rho_points3 <= radius_opt,
                       rho_points4 <= radius_opt)
    )


def find_n_cells_inside_optic_nerve(n, cell_corners, radius_opt):
    idxs_cells_inside_optic_nerve = np.where(find_cells_inside_optic_nerve(cell_corners, radius_opt))[0]
    if idxs_cells_inside_optic_nerve.size == 0:
        raise Exception('No cell can be selected.')
    idxs_n_cells_inside_optic_nerve = np.random.choice(idxs_cells_inside_optic_nerve, size=n, replace=False)
    return idxs_n_cells_inside_optic_nerve

