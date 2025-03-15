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
'''

import numpy as np
from scipy import signal

from helpers.fft_spectrum import *


class DopplerAlgo:
    """Compute Range-Doppler map"""

    def __init__(self, num_samples: int, num_chirps_per_frame: int, num_ant: int, mti_alpha: float = 0.8):
        """Create Range-Doppler map object

        Parameters:
            - num_samples:          Number of samples in a single chirp
            - num_chirps_per_frame: Number of chirp repetitions within a measurement frame
            - num_ant:              Number of antennas
            - mti_alpha:            Parameter alpha of Moving Target Indicator
        """
        self.num_chirps_per_frame = num_chirps_per_frame

        # compute Blackman-Harris Window matrix over chirp samples(range)
        try:
            self.range_window = signal.blackmanharris(num_samples).reshape(1, num_samples)
        except AttributeError:
            self.range_window = signal.windows.blackmanharris(num_samples).reshape(1, num_samples)

        # compute Blackman-Harris Window matrix over number of chirps(velocity)
        try:
            self.doppler_window = signal.blackmanharris(self.num_chirps_per_frame).reshape(1, self.num_chirps_per_frame)
        except AttributeError:
            self.doppler_window = signal.windows.blackmanharris(self.num_chirps_per_frame).reshape(1, self.num_chirps_per_frame)
            
        # parameter for moving target indicator (MTI)
        self.mti_alpha = mti_alpha

        # initialize MTI filter
        self.mti_history = np.zeros((self.num_chirps_per_frame, num_samples, num_ant))

    def compute_doppler_map(self, data: np.ndarray, i_ant: int):
        """Compute Range-Doppler map for i-th antennas

        Parameter:
            - data:     Raw-data for one antenna (dimension:
                        num_chirps_per_frame x num_samples)
            - i_ant:    RX antenna index
        """
        # Step 1 - Remove average from signal (mean removal)
        data = data - np.average(data)

        # Step 2 - MTI processing to remove static objects
        data_mti = data - self.mti_history[:, :, i_ant]
        self.mti_history[:, :, i_ant] = data * self.mti_alpha + self.mti_history[:, :, i_ant] * (1 - self.mti_alpha)

        # Step 3 - calculate fft spectrum for the frame
        fft1d = fft_spectrum(data_mti, self.range_window)

        # prepare for doppler FFT

        # Transpose
        # Distance is now indicated on y axis
        fft1d = np.transpose(fft1d)

        # Step 4 - Windowing the Data in doppler
        fft1d = np.multiply(fft1d, self.doppler_window)

        zp2 = np.pad(fft1d, ((0, 0), (0, self.num_chirps_per_frame)), "constant")
        fft2d = np.fft.fft(zp2) / self.num_chirps_per_frame

        # re-arrange fft result for zero speed at centre
        return np.fft.fftshift(fft2d, (1,))
'''
# New DopplerAlgo.py



# DopplerAlgo.py

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

from helpers.fft_spectrum import *


class DopplerAlgo:
    """Compute Range-Doppler map and extract velocity"""

    def __init__(self, num_samples: int, num_chirps_per_frame: int, num_ant: int, mti_alpha: float = 0.8):
        """Create Range-Doppler map object

        Parameters:
            - num_samples:           Number of samples in a single chirp
            - num_chirps_per_frame: Number of chirp repetitions within a measurement frame
            - num_ant:              Number of antennas
            - mti_alpha:            Parameter alpha of Moving Target Indicator
        """
        self.num_chirps_per_frame = num_chirps_per_frame

        # Compute Blackman-Harris Window matrix over chirp samples (range)
        try:
            self.range_window = signal.blackmanharris(num_samples).reshape(1, num_samples)
        except AttributeError:
            self.range_window = signal.windows.blackmanharris(num_samples).reshape(1, num_samples)

        # Compute Blackman-Harris Window matrix over number of chirps (velocity)
        try:
            self.doppler_window = signal.blackmanharris(self.num_chirps_per_frame).reshape(1, self.num_chirps_per_frame)
        except AttributeError:
            self.doppler_window = signal.windows.blackmanharris(self.num_chirps_per_frame).reshape(1, self.num_chirps_per_frame)
            
        # Parameter for moving target indicator (MTI)
        self.mti_alpha = mti_alpha

        # Initialize MTI filter
        self.mti_history = np.zeros((self.num_chirps_per_frame, num_samples, num_ant))

        # Doppler resolution (to be set based on radar parameters)
        self.doppler_resolution = None  # Hz per Doppler bin

    def compute_doppler_map(self, data: np.ndarray, i_ant: int):
        """Compute Range-Doppler map for the i-th antenna

        Parameters:
            - data:  Raw-data for one antenna (dimension: num_chirps_per_frame x num_samples)
            - i_ant: RX antenna index

        Returns:
            - fft2d_shifted: 2D FFT result shifted for zero velocity at center
        """
        # Step 1 - Remove average from signal (mean removal)
        data = data - np.average(data)

        # Step 2 - MTI processing to remove static objects
        data_mti = data - self.mti_history[:, :, i_ant]
        self.mti_history[:, :, i_ant] = data * self.mti_alpha + self.mti_history[:, :, i_ant] * (1 - self.mti_alpha)

        # Step 3 - Calculate FFT spectrum for the frame (Range FFT)
        fft1d = fft_spectrum(data_mti, self.range_window)

        # Transpose: Distance is now indicated on y-axis
        fft1d = np.transpose(fft1d)

        # Step 4 - Windowing the Data in Doppler
        fft1d = np.multiply(fft1d, self.doppler_window)

        # Zero-padding for Doppler FFT
        zp2 = np.pad(fft1d, ((0, 0), (0, self.num_chirps_per_frame)), "constant")
        fft2d = np.fft.fft(zp2) / self.num_chirps_per_frame

        # Re-arrange FFT result for zero speed at center
        fft2d_shifted = np.fft.fftshift(fft2d, axes=(1,))

        return fft2d_shifted

    def compute_velocity(self, data: np.ndarray, i_ant: int):
        """Compute velocity from Doppler map for the i-th antenna

        Parameters:
            - data:  Raw-data for one antenna (dimension: num_chirps_per_frame x num_samples)
            - i_ant: RX antenna index

        Returns:
            - velocity: Calculated velocity in m/s
            - doppler_map: Computed Doppler map (2D FFT result)
        """
        if self.doppler_resolution is None:
            raise ValueError("Doppler resolution not set. Please set doppler_resolution using set_doppler_resolution().")

        # Compute Doppler map
        doppler_map = self.compute_doppler_map(data, i_ant)

        # Compute Doppler spectrum by averaging over range bins
        doppler_spectrum = np.mean(np.abs(doppler_map), axis=0)

        # Find the peak in the Doppler spectrum
        peak_idx = np.argmax(doppler_spectrum)

        # Calculate Doppler bin index relative to zero velocity
        half_frame = self.num_chirps_per_frame // 2
        doppler_bin = peak_idx - half_frame

        # Compute velocity
        velocity = doppler_bin * self.doppler_resolution

        return velocity, doppler_map

    def set_doppler_resolution(self, chirp_duration: float, wavelength: float):
        """Set Doppler resolution based on chirp duration and wavelength

        Parameters:
            - chirp_duration: Duration of one chirp in seconds
            - wavelength: Wavelength of the radar signal in meters
        """
        total_observation_time = chirp_duration * self.num_chirps_per_frame
        doppler_frequency_resolution = 1 / total_observation_time  # Hz per Doppler bin
        self.doppler_resolution = doppler_frequency_resolution * (wavelength / 2)  # m/s per Doppler bin
