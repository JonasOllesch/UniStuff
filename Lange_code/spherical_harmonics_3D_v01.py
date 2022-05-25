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

plot_path = "./sph"

N_grid = 250
cmap = cm.hsv
dfunc_r = lambda x: np.abs(np.real(x))
dfunc_c = dfunc_r
dfunc_c = lambda x: np.angle(x)

n_phase_steps = 10
phases = np.linspace(0, 2 * np.pi, n_phase_steps, endpoint=False)

# complex-valued spherical harmonics
csph_l = 10
csph_m = 5

localphase = 0
sph_harm_coeffs = [[1, csph_l, csph_m, localphase, False]]
    
print("Preparing coordinate grid.")
theta = np.linspace(0, np.pi, N_grid)
phi = np.linspace(0, 2 * np.pi, N_grid)
theta, phi = np.meshgrid(theta, phi)

x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)
xyz = np.c_[x.ravel(), y.ravel(), z.ravel()]

def generate_lc(sph_harm_coeffs):
    ylm_lc = np.zeros(phi.shape, dtype = complex)
    for idx, [A, l, m, phase, conj] in enumerate(sph_harm_coeffs):
        if conj:
            ylm = np.conj(sph_harm(m, l, phi, theta))
        else:
            ylm = sph_harm(m, l, phi, theta)
        ylm_lc += A * ylm * np.exp(1j * phase)
    return ylm_lc


print("Calculating spherical harmonics.")
ylm_lc = generate_lc(sph_harm_coeffs)

print("Plotting.")
fig = plt.figure(figsize = figsize)

def plot_ylm(ylm_lc, phase):
    
    fcolors = dfunc_c(ylm_lc * np.exp(1j * phase)) 
    fmax, fmin = fcolors.max(), fcolors.min()
    if fmax != fmin:
        fcolors = (fcolors - fmin) / (fmax - fmin)
    r0 = dfunc_r(ylm_lc * np.exp(1j * phase))
    
    # Set the aspect ratio to 1 so our sphere looks spherical
    fig.clear()
    ax = fig.add_axes([0.025, 0.10, 0.9, 0.9], projection="3d")
    
    p = ax.plot_surface(x * r0, y * r0, z * r0, rstride=1, cstride=1, facecolors=cmap(fcolors))
    ax.set_box_aspect((1, 1, 1))
    xmax = ymax = zmax = np.max([r0 * x, r0 * y, r0 * z])
    xmin = ymin = zmin = np.min([r0 * x, r0 * y, r0 * z])
    xmax = ymax = zmax = np.max([xmax, ymax, zmax])
    xmin = ymin = zmin = np.min([xmin, ymin, zmin])
    
    cbar = fig.colorbar(
        mappable=None,
        cmap=cmap,
        norm=colors.Normalize(vmin=fmin, vmax=fmax),
        pad=0.12,
        shrink=0.65,
    )
    cbar.set_label(rf"$arg(Y_{{{csph_l},{csph_m}}})$")
    cbar.set_ticks(np.linspace(fmin, fmax, 3))
    cbar.set_ticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel("x", labelpad=15)
    ax.set_ylabel("y", labelpad=15)
    ax.set_zlabel("z", labelpad=15)

def plot_frame(index):
    plot_ylm(ylm_lc, phases[index])
  

if not os.path.exists(plot_path):
    os.makedirs(plot_path)
plot_frame(0)
plt.savefig(os.path.join(plot_path, "spherical_harmonic_l={:d}_m={:d}.png".format(csph_l, csph_m)), dpi = dpi)
    