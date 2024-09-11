import matplotlib.pyplot as plt
import numpy as np

import optic_geom as og
from settings import (
    scene_width_px,
    scene_height_px,
    r_ret,
    r_opt,
    radius_retina_sphere,
    n_fibers_theor,
)

np.random.seed(0)

# create rgc locations
locs_rgc_ret, locs_rgc_opt, locs_rgc_rf = og.sample_rgc_population(
    n_fibers_theor=n_fibers_theor,
    radius_retina=r_ret, 
    radius_optic_nerve=r_opt, 
)

# create receptive field matrix
rf_matrix = og.compute_rf_matrix(
    locs_rgc_rf=locs_rgc_rf,
    scene_width_px=scene_width_px,
    scene_height_px=scene_height_px,
    radius_retina=r_ret,
    radius_retina_sphere=radius_retina_sphere,
)

# save variables
og.save_optic_geom_variables(
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

# plot locations
fig, ax = plt.subplots(1, 3)
og.plot_rgcs(locs_rgc=locs_rgc_rf, ax=ax[0])
ax[0].set_title('RF centers')
og.plot_rgcs(locs_rgc=locs_rgc_ret, ax=ax[1])
ax[1].set_title('Soma locations')
og.plot_rgcs(locs_rgc=locs_rgc_opt, ax=ax[2])
ax[2].set_title('Fiber locations')
for i in range(3):
    ax[i].set_aspect('equal')
plt.show()
