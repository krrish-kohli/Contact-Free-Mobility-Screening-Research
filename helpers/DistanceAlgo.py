# ===========================================================================
# Copyright (C) 2021-2022 Infineon Technologies AG
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
from scipy import signal, constants as C
from .fft_spectrum import fft_spectrum

class DistanceAlgo:
    def __init__(self, chirp, num_chirps_per_frame: int):
        self.num_chirps_per_frame = num_chirps_per_frame

        # Range window
        try:
            self.range_window = signal.blackmanharris(chirp.num_samples).reshape(1, chirp.num_samples)
        except AttributeError:
            self.range_window = signal.windows.blackmanharris(chirp.num_samples).reshape(1, chirp.num_samples)

        # Robust meters-per-bin using slope S and actual Nfft
        Fs = chirp.sample_rate_Hz
        N = chirp.num_samples
        Nfft = 2 * N                              # we zero-pad by +N in fft_spectrum
        T = N / Fs
        B = abs(chirp.end_frequency_Hz - chirp.start_frequency_Hz)
        S = B / T                                  # Hz/s (chirp slope)
        delta_f = Fs / Nfft                        # Hz per FFT bin
        self.range_bin_length = C.c * delta_f / (2.0 * S)  # meters per displayed bin

        # Derive skip from HPF (minimum beat frequency admitted)
        hp = getattr(chirp, "hp_cutoff_Hz", 0.0)
        Rmin = (C.c * hp) / (2.0 * S) if hp > 0 else 0.0
        self.skip_bins = int(np.ceil(Rmin / self.range_bin_length))

    def compute_distance(self, chirp_data: np.ndarray):
        """
        chirp_data shape: [num_chirps_per_frame x num_samples]
        Returns: (peak_distance_m, distance_spectrum_per_bin)
        """
        range_fft = fft_spectrum(chirp_data, self.range_window)  # [M x N]
        # Incoherent sum across chirps
        distance_data = np.abs(range_fft).sum(axis=0) / self.num_chirps_per_frame

        # Skip bins we know the analog HPF will suppress
        idx = int(np.argmax(distance_data[self.skip_bins:])) + self.skip_bins
        peak_m = idx * self.range_bin_length
        return peak_m, distance_data
