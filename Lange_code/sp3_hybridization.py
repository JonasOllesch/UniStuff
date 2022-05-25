# Copyright 2022 Christoph Lange, TU Dortmund University.
# Published under the BSD-3 license.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# DESCRIPTION
# Calculates the time evolution of an arbitrarily shaped wave packet for an
# arbitrarily shaped potential, on a one-dimensional spatial grid. The results
# are shown either as a preview, or written to a video file.

import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys, os
from matplotlib import rc

rc('font', **{'size': 24})
rc('text', usetex=True)
dpi = 200
imgx = 1920
imgy = 1080
figsize=(imgx/dpi, imgy/dpi)

plot_path = "./sp3_hybridization"

N_grid = 250
cmap = cm.jet
dfunc = lambda x: np.abs(x)



comment = "1"
c_s, c_px, c_py, c_pz = 1, 1, 1, 1

comment = "2"
c_s, c_px, c_py, c_pz = 1, 1, -1, -1

comment = "3"
c_s, c_px, c_py, c_pz = 1, -1, -1, 1

comment = "4"
c_s, c_px, c_py, c_pz = 1, -1, 1, -1


rsph_l = 1
rsph_m = -1
sph_harm_coeffs_px = [
    [c_px * (-1) ** rsph_m / np.sqrt(2), rsph_l, rsph_m, False],
    [c_px * (-1) ** rsph_m / np.sqrt(2), rsph_l, rsph_m, True],
]

rsph_l = 1
rsph_m = 0 
sph_harm_coeffs_pz = [[c_pz, rsph_l, rsph_m, False]]

rsph_l = 1
rsph_m = 1
sph_harm_coeffs_py = [
    [c_py * (-1) ** rsph_m / (1j * np.sqrt(2)), rsph_l, -rsph_m, False],
    [c_py * -((-1) ** rsph_m) / (1j * np.sqrt(2)), rsph_l, -rsph_m, True],
]

rsph_l = 0
rsph_m = 0
sph_harm_coeffs_s = [[c_s, rsph_l, rsph_m, False]]

sph_harm_coeffs = (sph_harm_coeffs_s + sph_harm_coeffs_px + 
                   sph_harm_coeffs_py + sph_harm_coeffs_pz)


print("Preparing coordinate grid.")
theta = np.linspace(0, np.pi, N_grid)
phi = np.linspace(0, 2 * np.pi, N_grid)
theta, phi = np.meshgrid(theta, phi)

x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)
xyz = np.c_[x.ravel(), y.ravel(), z.ravel()]


def generate_lc(sph_harm_coeffs):
    ylm_lc = np.zeros(phi.shape, dtype=complex)
    for idx, sph_harm_coeff in enumerate(sph_harm_coeffs):
        l, m, conj = sph_harm_coeff[1:4]
        if conj:
            ylm = np.conj(sph_harm(m, l, phi, theta))
        else:
            ylm = sph_harm(m, l, phi, theta)
        ylm_lc += sph_harm_coeff[0] * ylm
    return ylm_lc


print("Calculating spherical harmonics.")
ylm_lc = generate_lc(sph_harm_coeffs)

print("Plotting.")


def plot_ylm(ylm_lc):
    fcolors = dfunc(ylm_lc)
    fmax, fmin = fcolors.max(), fcolors.min()
    if fmax != fmin: 
        fcolors = (fcolors - fmin) / (fmax - fmin)
    r0 = dfunc(ylm_lc)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([-0.1, 0.0, 1.0, 1.0], projection="3d")

    p = ax.plot_surface(
        x * r0, y * r0, z * r0, rstride=1, cstride=1, facecolors=cmap(fcolors)
    )
    ax.set_box_aspect((1, 1, 1))
    # xmax = ymax = zmax = np.max([r0 * x, r0 * y, r0 * z])
    # xmin = ymin = zmin = np.min([r0 * x, r0 * y, r0 * z])
    # xmax = ymax = zmax = np.max([xmax, ymax, zmax])
    # xmin = ymin = zmin = np.min([xmin, ymin, zmin])
    limit = np.max(np.abs(np.concatenate((r0 * x, r0 * y, r0 * z))))
    xmax = ymax = zmax = limit    
    xmin = ymin = zmin = -limit

    cbar = fig.colorbar(
        mappable=None,
        cmap=cmap,
        norm=colors.Normalize(vmin=fmin, vmax=fmax),
        pad=0.05,
        shrink=0.65,
    )
    cbar.set_label(r"$r = \psi_{" + "{:s}".format(comment) + r"}(\theta, \phi)$")
    cbar.set_ticks(np.linspace(fmin, fmax, 3))
    cbar.set_ticklabels([0, 0.5, 1])

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel(r"$x$", labelpad=-15)
    ax.set_ylabel(r"$y$", labelpad=-15)
    ax.set_zlabel(r"$z$", labelpad=-15)


if not os.path.exists(plot_path):
    os.makedirs(plot_path)
plot_ylm(ylm_lc)
plt.savefig(
    os.path.join(
        plot_path,
        "sp3_hybrid_{:s}.png".format(comment),
    ),
    dpi=dpi,
    # bbox_inches='tight',
)
