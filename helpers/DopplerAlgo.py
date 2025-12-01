# ===========================================================================
# Copyright (C) 2022 Infineon Technologies AG
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
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
# ===========================================================================

import numpy as np
from scipy import signal
from .fft_spectrum import fft_spectrum  # adjust if you use absolute imports

class DopplerAlgo:
    """
    Builds a range–Doppler map and extracts velocity.
    Correct scaling uses PRI (start-to-start between chirps) and Ndop bins.
    """

    def __init__(self, num_samples: int, num_chirps_per_frame: int, num_ant: int, mti_alpha: float = 0.8):
        self.num_chirps_per_frame = num_chirps_per_frame
        self.num_samples = num_samples
        self.num_ant = num_ant

        # Windows
        try:
            self.range_window = signal.blackmanharris(num_samples).reshape(1, num_samples)
        except AttributeError:
            self.range_window = signal.windows.blackmanharris(num_samples).reshape(1, num_samples)
        try:
            self.doppler_window = signal.blackmanharris(self.num_chirps_per_frame).reshape(1, self.num_chirps_per_frame)
        except AttributeError:
            self.doppler_window = signal.windows.blackmanharris(self.num_chirps_per_frame).reshape(1, self.num_chirps_per_frame)

        # Simple MTI (leaky integrator per antenna)
        self.mti_alpha = mti_alpha
        self.mti_history = np.zeros((self.num_chirps_per_frame, num_samples, num_ant), dtype=float)

        # Velocity scaling (set via set_doppler_scaling)
        self._v_per_bin = None

    def set_doppler_scaling(self, pri_s: float, wavelength_m: float, ndop_bins: int):
        """
        Configure meters-per-second per Doppler bin:
        v_per_bin = (PRF / Ndop) * (λ/2)
        where PRF = 1/PRI and Ndop is the Doppler FFT length after zero-padding.
        """
        prf = 1.0 / pri_s
        self._v_per_bin = (prf / ndop_bins) * (wavelength_m / 2.0)

    def compute_doppler_map(self, data: np.ndarray, i_ant: int):
        """
        data shape: [num_chirps_per_frame, num_samples]
        returns complex map of shape [num_ranges, Ndop]
        """
        # Mean removal + MTI
        data = data - np.average(data)
        data_mti = data - self.mti_history[:, :, i_ant]
        self.mti_history[:, :, i_ant] = (
            data * self.mti_alpha + self.mti_history[:, :, i_ant] * (1.0 - self.mti_alpha)
        )

        # Range FFT along fast-time (per chirp)
        fft1d = fft_spectrum(data_mti, self.range_window)   # shape: [M x N]
        fft1d = np.transpose(fft1d)                         # [N ranges x M chirps]

        # Doppler FFT along slow-time, with 2x zero-padding
        M = self.num_chirps_per_frame
        Ndop = 2 * M
        fft1d = np.multiply(fft1d, self.doppler_window)     # window across chirps
        zp2 = np.pad(fft1d, ((0, 0), (0, M)), mode="constant")
        doppler = np.fft.fft(zp2, n=Ndop, axis=1) / Ndop
        doppler = np.fft.fftshift(doppler, axes=(1,))       # center zero Doppler
        return doppler

    def compute_velocity(self, data, i_ant, range_bin=None, gate_half_width=2):
        """
        Optionally gate the Doppler computation to a range bin or a small window around it.
        """
        if self._v_per_bin is None:
            raise ValueError("Call set_doppler_scaling(pri_s, wavelength_m, ndop_bins) before compute_velocity().")

        doppler_map = self.compute_doppler_map(data, i_ant)
        mag = np.abs(doppler_map)

        if range_bin is not None:
            lo = max(0, range_bin - gate_half_width)
            hi = min(mag.shape[0], range_bin + gate_half_width + 1)
            doppler_spectrum = mag[lo:hi, :].mean(axis=0)
        else:
            doppler_spectrum = mag.mean(axis=0)

        peak_idx = int(np.argmax(doppler_spectrum))
        Ndop = mag.shape[1]
        zero = Ndop // 2                     # correct center after fftshift
        doppler_bin = peak_idx - zero

        velocity_m_s = doppler_bin * self._v_per_bin
        return velocity_m_s, doppler_map
