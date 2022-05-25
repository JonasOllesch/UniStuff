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
import matplotlib.pyplot as plt
from sympy.physics.hydrogen import Psi_nlm
from sympy import Symbol, lambdify
from matplotlib import rc


rc("font", **{"size": 12})
rc("text", usetex=True)
rc("animation", html="html5")
rc("figure", figsize = (5, 4.2))

r = Symbol("r", positive=True)
phi = Symbol("phi", real=True)
theta = Symbol("theta", real=True)

fmt = lambda x: "{:.0f}%".format(x)

points = 1001
hbar = 1.05457182e-34
ryd = 2.179872e-18 # Rydberg energy in Joules

# Superposition coefficients, (A, n, l, m). A: Amplitude.
# Enable _one_ of the following configurations.

# c_wf = [[1, 1, 0, 0], [1, 2, 1, 1]]
# comment = "spos1"
# max_r = 10

c_wf = [[4, 2, 1, 0], [5, 3, 2, 1]]
comment = "spos2"
max_r = 20

lin_scale = np.linspace(-max_r, max_r, points)
xgrid, ygrid = np.meshgrid(lin_scale, lin_scale)
rgrid, phigrid = np.sqrt(xgrid**2 + ygrid**2), np.arctan2(ygrid, xgrid)

wf_lamb = [lambdify([r, phi], Psi_nlm(n, l, m, r, phi, np.pi/2, 1), "numpy") for [A, n, l, m] in c_wf]
wf_num_stat = [wfl(rgrid, phigrid) for wfl in wf_lamb]

def E_nl(n):
    return -ryd / n**2

def plot_wf_real(t, dfunc = np.real, cmap = "seismic", symdlim = True):
    wavefunc_num = np.zeros_like(rgrid, dtype = complex)
    for idx in range(len(c_wf)):
        wavefunc_num += c_wf[idx][0] * np.exp(-1j * E_nl(c_wf[idx][1]) * t / hbar) * wf_num_stat[idx]
    wavefunc_num = dfunc(wavefunc_num)
    dlim = max(np.min(wavefunc_num), np.max(wavefunc_num))
    if symdlim: 
        clim = (-dlim, dlim)
    else:
        clim = (0, dlim)

    fig = plt.figure(constrained_layout=True)
    ax = fig.add_axes([0.15, 0.13, 0.8, 0.8])
    im = ax.imshow(wavefunc_num, cmap = cmap, aspect = "1.0", clim = clim)
    ticker = [0, int(points / 4), int(points / 2), int(3 * points / 4), points - 1]
    labeler = [
        lin_scale[0],
        lin_scale[int(points / 4)],
        lin_scale[int(points / 2)],
        lin_scale[int(3 * points / 4)],
        lin_scale[-1],
    ]
    ax.set_xticks(ticker, [fmt(i) for i in labeler])
    ax.set_yticks(ticker, [fmt(i) for i in labeler[::-1]])
    ax.set_xlabel("$x\,/\,a_0$")
    ax.set_ylabel("$y\,/\,a_0$")
    title = "[" + ", ".join(["({:d},{:d},{:d})".format(n, l, m) for (A, n, l, m) in c_wf]) + "]"
    title += r",  $t = $" + " {:.3f} fs".format(t * 1e15)
    ax.set_title(title)
    cbar = plt.colorbar(im, ax = ax)
    cbar.set_label(r"$\mathcal{R}e \Psi(r, \phi, \theta = 90°)$")
    cbar.set_ticks(np.linspace(-dlim, dlim, 3))
    cbar.set_ticklabels([r"$-$", r"0", r"$+$"])
    
def plot_wf_abs2(t, dfunc = lambda x: np.abs(x)**2, cmap = "afmhot_r", symdlim = False):

    wavefunc_num = np.zeros_like(rgrid, dtype = complex)

    for idx in range(len(c_wf)):
        wavefunc_num += c_wf[idx][0] * np.exp(-1j * E_nl(c_wf[idx][1]) * t / hbar) * wf_num_stat[idx]
    wavefunc_num = dfunc(wavefunc_num)
    dlim = max(np.min(wavefunc_num), np.max(wavefunc_num))
    
    if symdlim: 
        clim = (-dlim, dlim)
    else:
        clim = (0, dlim)

    fig = plt.figure(constrained_layout=True)
    ax = fig.add_axes([0.15, 0.13, 0.8, 0.8])
    im = ax.imshow(wavefunc_num, cmap = cmap, aspect = "1.0", clim = clim)
    ticker = [0, int(points / 4), int(points / 2), int(3 * points / 4), points - 1]
    labeler = [
        lin_scale[0],
        lin_scale[int(points / 4)],
        lin_scale[int(points / 2)],
        lin_scale[int(3 * points / 4)],
        lin_scale[-1],
    ]

    ax.set_xticks(ticker, [fmt(i) for i in labeler])
    ax.set_yticks(ticker, [fmt(i) for i in labeler[::-1]])

    ax.set_xlabel("$x\,/\,a_0$")
    ax.set_ylabel("$y\,/\,a_0$")

    title = "[" + ", ".join(["({:d},{:d},{:d})".format(n, l, m) for (A, n, l, m) in c_wf]) + "]"
    title += r",  $t = $" + " {:.3f} fs".format(t * 1e15)

    ax.set_title(title)

    cbar = plt.colorbar(im, ax = ax)
    cbar.set_label(r"$|\Psi(r, \phi, \theta = 90°)|^2$")
    cbar.set_ticks(np.linspace(0, dlim, 2))
    cbar.set_ticklabels([r"0", r"$+$"])

plot_wf_abs2(0e-15)
plt.savefig("Abs2_psi_{:s}.png".format(comment), dpi = 300)
plot_wf_real(0e-15)
plt.savefig("Re_psi_{:s}.png".format(comment), dpi = 300)
