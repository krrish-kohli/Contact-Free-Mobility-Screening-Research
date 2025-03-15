import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


# -------------------------------------------------
# Presentation
# -------------------------------------------------
class Draw:
    # Represents drawing for example
    #
    # Draw is done for each antenna, and each antenna is represented for
    # other subplot

    def __init__(self, max_speed_m_s, max_range_m, num_ant):
        # max_range_m:   maximum supported range
        # max_speed_m_s: maximum supported speed
        # num_ant:      number of available antennas
        self._h = []
        self._max_speed_m_s = max_speed_m_s
        self._max_range_m = max_range_m
        self._num_ant = num_ant

        plt.ion()

        self._fig, ax = plt.subplots(nrows=1, ncols=num_ant, figsize=((num_ant + 1) // 2, 2))
        if (num_ant == 1):
            self._ax = [ax]
        else:
            self._ax = ax

        self._fig.canvas.manager.set_window_title("Doppler")
        self._fig.set_size_inches(3 * num_ant + 1, 3 + 1 / num_ant)
        self._fig.canvas.mpl_connect('close_event', self.close)
        self._is_window_open = True

    def _draw_first_time(self, data_all_antennas):
        # First time draw
        #
        # It computes minimal, maximum value and draw data for all antennas
        # in same scale
        # data_all_antennas: array of raw data for each antenna

        minmin = min([np.min(data) for data in data_all_antennas])
        maxmax = max([np.max(data) for data in data_all_antennas])

        for i_ant in range(self._num_ant):
            data = data_all_antennas[i_ant]
            h = self._ax[i_ant].imshow(
                data,
                vmin=minmin, vmax=maxmax,
                cmap='hot',
                extent=(-self._max_speed_m_s,
                        self._max_speed_m_s,
                        0,
                        self._max_range_m),
                origin='lower')
            self._h.append(h)

            self._ax[i_ant].set_xlabel("velocity (m/s)")
            self._ax[i_ant].set_ylabel("distance (m)")
            self._ax[i_ant].set_title("antenna #" + str(i_ant))
        self._fig.subplots_adjust(right=0.8)
        cbar_ax = self._fig.add_axes([0.85, 0.0, 0.03, 1])

        cbar = self._fig.colorbar(self._h[0], cax=cbar_ax)
        cbar.ax.set_ylabel("magnitude (dB)")

    def _draw_next_time(self, data_all_antennas):
        # data_all_antennas: array of raw data for each antenna

        for i_ant in range(0, self._num_ant):
            data = data_all_antennas[i_ant]
            self._h[i_ant].set_data(data)

    def draw(self, data_all_antennas):
        # Draw plots for all antennas
        # data_all_antennas: array of raw data for each antenna
        if self._is_window_open:
            if len(self._h) == 0:  # handle the first run
                self._draw_first_time(data_all_antennas)
            else:
                self._draw_next_time(data_all_antennas)

            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()

    def close(self, event=None):
        if self.is_open():
            self._is_window_open = False
            plt.close(self._fig)
            plt.close('all')  # Needed for Matplotlib ver: 3.4.0 and 3.4.1
            print('Application closed!')

    def is_open(self):
        return self._is_window_open


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def parse_program_arguments(description):
    # Parse all program attributes
    # description:   describes program

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-n', '--nframes', type=int,
                        default=None, help="number of frames to process (default: all frames)")
    parser.add_argument('-f', '--file', type=str, required=True,
                        help="path to .npy file containing the data")
    parser.add_argument('--max_range_m', type=float, default=4.8,
                        help="maximum supported range in meters (default: 4.8)")
    parser.add_argument('--max_speed_m_s', type=float, default=2.45,
                        help="maximum supported speed in m/s (default: 2.45)")

    return parser.parse_args()


def linear_to_dB(x):
    return 20 * np.log10(abs(x))


def fft_spectrum(data, window):
    """
    Compute FFT spectrum with windowing.

    Parameters:
        data: 2D numpy array, where FFT is computed along the last axis
        window: 1D or 2D numpy array to be applied element-wise
    """
    # Apply window
    data_windowed = data * window
    # Perform FFT along last axis
    fft_result = np.fft.fft(data_windowed, axis=-1)
    # Return FFT result
    return fft_result


# -------------------------------------------------
# DopplerAlgo class
# -------------------------------------------------
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
            self.doppler_window = signal.windows.blackmanharris(self.num_chirps_per_frame).reshape(1,
                                                                                                   self.num_chirps_per_frame)

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
        fft2d = np.fft.fft(zp2, axis=1) / self.num_chirps_per_frame

        # re-arrange fft result for zero speed at centre
        return np.fft.fftshift(fft2d, (1,))


# -------------------------------------------------
# Main logic
# -------------------------------------------------
if __name__ == '__main__':
    args = parse_program_arguments(
        '''Displays range doppler map from Radar Data stored in .npy file''')

    data = np.load(args.file)

    if data.ndim == 4:
        # Data has shape (num_frames, num_antennas, num_chirps_per_frame, num_samples)
        num_frames, num_antennas, num_chirps_per_frame, num_samples = data.shape
    elif data.ndim == 3:
        # Data has shape (num_antennas, num_chirps_per_frame, num_samples)
        num_frames = 1
        data = data[np.newaxis, ...]  # Add frame dimension
        num_frames, num_antennas, num_chirps_per_frame, num_samples = data.shape
    else:
        raise ValueError("Invalid data shape: expected 3 or 4 dimensions")

    if args.nframes is None or args.nframes > num_frames:
        args.nframes = num_frames

    # Define metrics (e.g., max_range_m, max_speed_m_s)
    metrics = {
        'range_resolution_m': 0.15,
        'max_range_m': args.max_range_m,
        'max_speed_m_s': args.max_speed_m_s,
        'speed_resolution_m_s': 0.2,
        'center_frequency_Hz': 60_750_000_000,
    }

    doppler = DopplerAlgo(num_samples, num_chirps_per_frame, num_antennas)
    draw = Draw(metrics['max_speed_m_s'], metrics['max_range_m'], num_antennas)

    for frame_number in range(args.nframes):  # for each frame
        if not draw.is_open():
            break
        frame_data = data[frame_number]
        data_all_antennas = []
        for i_ant in range(0, num_antennas):  # for each antenna
            mat = frame_data[i_ant, :, :]
            dfft_dbfs = linear_to_dB(doppler.compute_doppler_map(mat, i_ant))
            data_all_antennas.append(dfft_dbfs)
        draw.draw(data_all_antennas)

    draw.close()
