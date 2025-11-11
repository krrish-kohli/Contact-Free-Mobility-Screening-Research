# ===========================================================================
# Copyright (C) 2021 Infineon Technologies AG
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

def fft_spectrum(mat: np.ndarray, range_window: np.ndarray) -> np.ndarray:
    """
    Range FFT for FMCW:
    - DC removal per chirp
    - window across fast-time
    - zero-pad to 2N
    - FFT, keep positive frequencies
    Returns shape: [num_chirps x num_samples]
    """
    num_chirps, num_samples = np.shape(mat)

    # DC removal per chirp (fast-time)
    avgs = np.average(mat, axis=1).reshape(num_chirps, 1)
    mat = mat - avgs

    # Apply window
    mat = np.multiply(mat, range_window)

    # Zero-pad to 2N and FFT
    N = num_samples
    Nfft = 2 * N
    zp = np.pad(mat, ((0, 0), (0, N)), mode="constant")
    rfft = np.fft.fft(zp, n=Nfft, axis=1) / Nfft

    # Keep positive frequencies; scale by 2 to fold negative side (amplitude convention)
    rfft = 2.0 * rfft[:, :N]
    return rfft
