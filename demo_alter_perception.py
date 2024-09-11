import matplotlib.pyplot as plt
import numpy as np

import ideal_stimulation as ideal_stim
import optic_geom as og
import modify_perception as mp

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

# loading scenes
idx_images = np.array([0, 1, 2, 6]) 
n_scenes = len(idx_images)
scenes = ideal_stim.TargetImagesLoadingClass.load_scenes(
    n_scenes=n_scenes,
    idx_images=idx_images,
    scene_width_px=scene_width_px,
    scene_height_px=scene_height_px
)

# creation cells for stimulation
L=[2, 2, 1, 1, 1]
vision_type = 'optic nerve'
(
    cell_corners, cell_centers, idxs_inside_cells, n_cells, locs_rgc 
)= ideal_stim.gen_cells(vision_type, L, locs_rgc_opt, locs_rgc_ret)
idxs_inside_cells = ideal_stim.compute_idxs_inside_cells(
    cell_corners=cell_corners,
    locs_rgc=locs_rgc
)
cell_centers = np.hstack((
    np.mean(cell_corners[:, 0:2], axis=1)[:, None],
    np.mean(cell_corners[:, 2:], axis=1)[:, None]
))

# initializing local and global shuffling
n_rgcs = len(locs_rgc)
n_swaps_local_shuffle = int(.2 * n_rgcs)  # parameter
n_swaps_global_shuffle = int(n_cells / 5)

# initializing damaged sites
n_off = int(n_cells / 10)
idxs_cells_off = mp.find_n_cells_inside_optic_nerve(n_off, cell_corners, r_opt)

# plot damaged sites
fig0, ax = plt.subplots(figsize=(5, 5))
ideal_stim.plot_grid(cell_corners=cell_corners, grid_color='k', idxs_cells_off=idxs_cells_off, ax=ax)
og.plot_rgcs(locs_rgc=locs_rgc_opt, ax=ax)
ax.set_axis_off()

# initializing perceptions lists
perceptions_original = [None for _ in range(n_scenes)]
perceptions_local_shuffle = [None for _ in range(n_scenes)]
perceptions_global_shuffle_uncompensated = [None for _ in range(n_scenes)]
perceptions_global_shuffle_compensated = [None for _ in range(n_scenes)]
perceptions_global_and_local_shuffle = [None for _ in range(n_scenes)]
perceptions_some_off = [None for _ in range(n_scenes)]
n_perceptions = 6

# generation of the perception in the different scenarios
for i_scene, scene in enumerate(scenes):
    perceptions_original[i_scene], _ = ideal_stim.compute_discretized_image(
        image=scene,
        rf_matrix=rf_matrix,
        idxs_inside_cells=idxs_inside_cells,
        scene_height_px=scene_height_px,
        scene_width_px=scene_width_px,
    )

    # "local" shuffle
    perceptions_local_shuffle[i_scene], rf_matrix_local_shuffle, \
    idxs_to_swap_1, idxs_to_swap_2 = \
        mp.local_shuffle(scene, rf_matrix, n_swaps_local_shuffle, 
                      locs_rgc, idxs_inside_cells, scene_height_px, 
                      scene_width_px, return_idxs=True)

    
    # "global" shuffle
    idxs_inside_cells_after_global_shuffle, cell_centers_global, n_cells_global, \
        locs_rgc_global_shuffled, idxs_inside_cells_global, idxs_outside_cells_global, \
        cell_centers_global_shuffled, _ = \
        mp.generate_idxs_inside_cells_after_global_shuffle(
            n_swaps_global_shuffle, locs_rgc, cell_corners
        )
    
    # global perception uncompensated
    perceptions_global_shuffle_uncompensated[i_scene], _ = ideal_stim.compute_discretized_image(
        image=scene,
        rf_matrix=rf_matrix,
        idxs_inside_cells=idxs_inside_cells,
        idxs_inside_cells_after_global_shuffle=idxs_inside_cells_after_global_shuffle,
        scene_height_px=scene_height_px,
        scene_width_px=scene_width_px,
    )

    # global perception compensated with map reconstruction
    perceptions_global_shuffle_compensated[i_scene], _ = ideal_stim.compute_discretized_image(
        image=scene,
        rf_matrix=rf_matrix,
        idxs_inside_cells=idxs_inside_cells_after_global_shuffle,
        scene_height_px=scene_height_px,
        scene_width_px=scene_width_px,
    )

    # global and local
    perceptions_global_and_local_shuffle[i_scene], _ = ideal_stim.compute_discretized_image(
        image=scene,
        rf_matrix=rf_matrix_local_shuffle,
        idxs_inside_cells=idxs_inside_cells_after_global_shuffle,
        scene_height_px=scene_height_px,
        scene_width_px=scene_width_px,
    )

    # some off from the global and local
    perceptions_some_off[i_scene], _, n_off = ideal_stim.compute_discretized_image(
        image=scene,
        rf_matrix=rf_matrix,
        idxs_inside_cells=idxs_inside_cells,
        scene_height_px=scene_height_px,
        scene_width_px=scene_width_px,
        some_off=True,
        idxs_cells_off=idxs_cells_off,
        return_n=True,
    )

# plot of the perceptions
n_row_plots = n_perceptions + 1
fig1, axs = plt.subplots(n_row_plots, n_scenes, figsize=(12, 18))
for i_scene, scene in enumerate(scenes):
    
    if i_scene == 0:

        axs[0, i_scene].set_ylabel('Scene', rotation=0, horizontalalignment='right', verticalalignment='center_baseline')
        axs[1, i_scene].set_ylabel('Original\nperception', rotation=0, horizontalalignment='right', verticalalignment='center_baseline')
        axs[2, i_scene].set_ylabel(f'Local shuffle',
                                   rotation=0, horizontalalignment='right', verticalalignment='center_baseline')
        axs[3, i_scene].set_ylabel(f'Uncorrected\nglobal shuffle',
                                   rotation=0, horizontalalignment='right', verticalalignment='center_baseline')
        axs[4, i_scene].set_ylabel(f'Corrected\nglobal shuffle',
                                   rotation=0, horizontalalignment='right', verticalalignment='center_baseline')
        axs[5, i_scene].set_ylabel(f'Local and corrected\nglobal shuffle',
                                   rotation=0, horizontalalignment='right', verticalalignment='center_baseline')
        axs[6, i_scene].set_ylabel(f'Damaged\nstimulation sites',
                                   rotation=0, horizontalalignment='right', verticalalignment='center_baseline')
        
    for j in range(n_row_plots):
        axs[j, i_scene].tick_params(axis='both', which='both', 
                   left=False, right=False, 
                   top=False, bottom=False, 
                   labelleft=False, labelright=False, 
                   labeltop=False, labelbottom=False)
        axs[j, i_scene].spines['left'].set_visible(False)
        axs[j, i_scene].spines['right'].set_visible(False)
        axs[j, i_scene].spines['top'].set_visible(False)
        axs[j, i_scene].spines['bottom'].set_visible(False)

    axs[0, i_scene].imshow(scene, cmap='gray')
    axs[1, i_scene].imshow(perceptions_original[i_scene], cmap='gray')
    axs[2, i_scene].imshow(perceptions_local_shuffle[i_scene], cmap='gray')
    axs[3, i_scene].imshow(perceptions_global_shuffle_uncompensated[i_scene], cmap='gray')
    axs[4, i_scene].imshow(perceptions_global_shuffle_compensated[i_scene], cmap='gray')
    axs[5, i_scene].imshow(perceptions_global_and_local_shuffle[i_scene], cmap='gray')
    axs[6, i_scene].imshow(perceptions_some_off[i_scene], cmap='gray')

locs_rgc_opt_local_shuffled = np.copy(locs_rgc_opt)
for idx_to_swap_1, idx_to_swap_2 in zip(idxs_to_swap_1, idxs_to_swap_2):
    tmp = locs_rgc_opt_local_shuffled[idx_to_swap_1]
    locs_rgc_opt_local_shuffled[idx_to_swap_1] = locs_rgc_opt_local_shuffled[idx_to_swap_2]
    locs_rgc_opt_local_shuffled[idx_to_swap_2] = tmp

# plot of the locally-shuffled locations of the RGCs
alpha = .5

fig3, ax = plt.subplots(1, 3, constrained_layout=True, figsize=(10, 5))
new_inferno = plt.get_cmap('hsv', n_rgcs)
colors = np.zeros((n_rgcs, 4))
for i in range(n_rgcs):
    colors[i, :] = new_inferno(i/n_rgcs)
locs_rgc_opt_rho = np.sqrt(locs_rgc_opt[:, 0]**2 + locs_rgc_opt[:, 1]**2)
idx_ordered = np.argsort(locs_rgc_opt_rho)
ax = ax.ravel()
ax[0].scatter(locs_rgc_rf[idx_ordered, 0], locs_rgc_rf[idx_ordered, 1], c=colors, alpha=alpha)
ax[1].scatter(locs_rgc_ret[idx_ordered, 0], locs_rgc_ret[idx_ordered, 1], c=colors, alpha=alpha)
ax[2].scatter(locs_rgc_opt_local_shuffled[idx_ordered, 0], locs_rgc_opt_local_shuffled[idx_ordered, 1], c=colors, alpha=alpha)
ax[0].title.set_text('RF centers')
ax[0].set_xlabel('X (RF) [a. u.]')
ax[0].set_ylabel('Y (RF) [a. u.]')
ax[1].title.set_text('Soma locations')
ax[2].title.set_text('Fiber locations')
ax[1].set_xlabel('X (retina) [mm]')
ax[1].set_ylabel('Y (retina) [mm]')
ax[2].set_xlabel('X (optic nerve) [mm]')
ax[2].set_ylabel('Y (optic nerve) [mm]')
for i in range(3):
    ax[i].set_aspect('equal')

fig4, ax = plt.subplots(1, 3, constrained_layout=True, figsize=(10, 5))
locs_rgc_opt_theta = np.arctan2(locs_rgc_opt[:, 1], locs_rgc_opt[:, 0])
idx_ordered = np.argsort(locs_rgc_opt_theta)
ax = ax.ravel()
ax[0].scatter(locs_rgc_rf[idx_ordered, 0], locs_rgc_rf[idx_ordered, 1], c=colors, alpha=alpha)
ax[1].scatter(locs_rgc_ret[idx_ordered, 0], locs_rgc_ret[idx_ordered, 1], c=colors, alpha=alpha)
ax[2].scatter(locs_rgc_opt_local_shuffled[idx_ordered, 0], locs_rgc_opt_local_shuffled[idx_ordered, 1], c=colors, alpha=alpha)
ax[0].title.set_text('RF centers')
ax[1].title.set_text('Soma locations')
ax[2].title.set_text('Fiber locations')
ax[0].set_xlabel('X (RF) [a. u.]')
ax[0].set_ylabel('Y (RF) [a. u.]')
ax[1].set_xlabel('X (retina) [mm]')
ax[1].set_ylabel('Y (retina) [mm]')
ax[2].set_xlabel('X (optic nerve) [mm]')
ax[2].set_ylabel('Y (optic nerve) [mm]')
for i in range(3):
    ax[i].set_aspect('equal')

plt.show()
