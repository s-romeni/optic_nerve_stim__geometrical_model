import matplotlib.pyplot as plt
import numpy as np

import ideal_stimulation as ideal_stim
import optic_geom as og

np.random.seed(0)

# load optic geom variables
(
    r_ret,
    r_opt,
    radius_retina_sphere,
    locs_rgc_ret, 
    locs_rgc_opt, 
    locs_rgc_rf,
    scene_width_px,
    scene_height_px,
    rf_matrix,
) = og.load_optic_geom_variables()

# load scene
n_ranges = 255
scene_filename = ".\\target_images\\stimulus1.npy"
target_scene = ideal_stim.load_target_image(
    scene_filename,
    scene_width_px=scene_width_px,
    scene_height_px=scene_height_px,
    n_ranges=n_ranges
)

# create perception
vision_type = 'optic nerve'
L = np.ones(16,).tolist()
(
    cell_corners, cell_centers, idxs_inside_cells, n_cells, _ 
) = ideal_stim.gen_cells(vision_type, L, locs_rgc_opt, locs_rgc_ret)
perception, _ = ideal_stim.compute_discretized_image(
    image=target_scene,
    rf_matrix=rf_matrix,
    idxs_inside_cells=idxs_inside_cells,
    scene_height_px=scene_height_px,
    scene_width_px=scene_width_px,
    n_ranges=n_ranges,
)

# plot
fig, axs = plt.subplots(1, 2)
axs[0].imshow(target_scene, cmap='gray')
axs[1].imshow(perception, cmap='gray')
axs[0].set_axis_off()
axs[1].set_axis_off()
plt.tight_layout()
plt.show()
