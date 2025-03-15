import matplotlib
matplotlib.use('TkAgg')
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ifxradarsdk import get_version_full
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwMetrics
from helpers.DopplerAlgo import *


class VelocityTrackerPlot:
    def __init__(self):
        plt.ion()
        self._fig, self._ax = plt.subplots()
        self._fig.canvas.manager.set_window_title("Object Velocity Over Time")
        self._ax.set_xlabel("Frame")
        self._ax.set_ylabel("Velocity (m/s)")
        self._ax.set_title("Object Velocity Over Time")
        self._velocity_line, = self._ax.plot([], [], 'b-o')
        self._velocities = []
        self._frames = []

    def update(self, frame_number, velocity):
        self._velocities.append(velocity)
        self._frames.append(frame_number)
        self._velocity_line.set_xdata(self._frames)
        self._velocity_line.set_ydata(self._velocities)
        self._ax.relim()
        self._ax.autoscale_view()

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def close(self):
        plt.close(self._fig)

    def get_data(self):
        """Return the collected frames and velocities."""
        return self._frames, self._velocities


def linear_to_dB(x):
    return 20 * np.log10(abs(x))


def get_user_input_gui(default_time=10, default_frate=5):
    # Create a small Tkinter window to get user input
    root = tk.Tk()
    root.title("Radar Configuration")

    record_time_var = tk.DoubleVar(value=default_time)
    frate_var = tk.IntVar(value=default_frate)

    frame = ttk.Frame(root, padding="10 10 10 10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    ttk.Label(frame, text="Recording time (seconds):").grid(row=0, column=0, sticky=tk.W)
    time_entry = ttk.Entry(frame, textvariable=record_time_var, width=10)
    time_entry.grid(row=0, column=1, sticky=(tk.W, tk.E))

    ttk.Label(frame, text="Frame rate (Hz):").grid(row=1, column=0, sticky=tk.W)
    frate_entry = ttk.Entry(frame, textvariable=frate_var, width=10)
    frate_entry.grid(row=1, column=1, sticky=(tk.W, tk.E))

    def on_start():
        root.quit()

    start_button = ttk.Button(frame, text="Start", command=on_start)
    start_button.grid(row=2, column=0, columnspan=2, pady=10)

    root.mainloop()
    recording_time = record_time_var.get()
    frate = frate_var.get()
    root.destroy()

    return recording_time, frate


if __name__ == '__main__':
    # Get user input from GUI
    recording_time, frate = get_user_input_gui(default_time=10, default_frate=5)
    nframes = int(recording_time * frate)

    # Adjust this path to your desired directory and filename
    output_path = r"C:\Users\Raj Patel\Desktop\radar_sdk\Data_Store\velocity_data.xlsx"

    with DeviceFmcw() as device:
        print(f"Radar SDK Version: {get_version_full()}")
        print("Sensor: " + str(device.get_sensor_type()))
        print(f"Recording for {recording_time} seconds at {frate} Hz, total {nframes} frames.")

        # use all available antennas
        num_rx_antennas = device.get_sensor_information()["num_rx_antennas"]

        metrics = FmcwMetrics(
            range_resolution_m=0.15,
            max_range_m=4.8,
            max_speed_m_s=2.45,
            speed_resolution_m_s=0.2,
            center_frequency_Hz=60_750_000_000,
        )

        # create acquisition sequence based on metrics parameters
        sequence = device.create_simple_sequence(FmcwSimpleSequenceConfig())
        sequence.loop.repetition_time_s = 1 / frate  # set frame repetition time

        # convert metrics into chirp loop parameters
        chirp_loop = sequence.loop.sub_sequence.contents
        device.sequence_from_metrics(metrics, chirp_loop)

        # set remaining chirp parameters which are not derived from metrics
        chirp = chirp_loop.loop.sub_sequence.contents.chirp
        chirp.sample_rate_Hz = 1_000_000
        chirp.rx_mask = (1 << num_rx_antennas) - 1
        chirp.tx_mask = 1
        chirp.tx_power_level = 31
        chirp.if_gain_dB = 33
        chirp.lp_cutoff_Hz = 500000
        chirp.hp_cutoff_Hz = 80000

        device.set_acquisition_sequence(sequence)

        doppler = DopplerAlgo(chirp.num_samples, chirp_loop.loop.num_repetitions, num_rx_antennas)

        vel_plot = VelocityTrackerPlot()

        for frame_number in range(nframes):
            frame_contents = device.get_next_frame()
            frame_data = frame_contents[0]
            data_all_antennas = []

            # Compute range-Doppler map for each antenna and convert to dB
            for i_ant in range(num_rx_antennas):
                mat = frame_data[i_ant, :, :]
                dfft_dbfs = linear_to_dB(doppler.compute_doppler_map(mat, i_ant))
                data_all_antennas.append(dfft_dbfs)

            # Use the first antenna data for object velocity estimation
            rd_map = data_all_antennas[0]

            # Create a doppler axis based on actual doppler bins of rd_map
            num_doppler_bins = rd_map.shape[1]
            doppler_axis = np.linspace(-metrics.max_speed_m_s, metrics.max_speed_m_s, num_doppler_bins)

            # Find the peak in the range-Doppler map
            max_idx = np.argmax(rd_map)
            range_idx, dop_idx = np.unravel_index(max_idx, rd_map.shape)

            # Convert doppler index to velocity
            object_velocity = doppler_axis[dop_idx]
            print(f"Frame {frame_number}: Object velocity = {object_velocity:.2f} m/s")

            # Update only the velocity vs. time plot
            vel_plot.update(frame_number, object_velocity)

        # Close the velocity plot
        vel_plot.close()

        # Retrieve the recorded data (frames and velocities)
        frames, velocities = vel_plot.get_data()

        # Create a DataFrame from the data
        df = pd.DataFrame({"Frame": frames, "Velocity (m/s)": velocities})

        # Save to Excel file
        df.to_excel(output_path, index=False)
        print(f"Data saved to {output_path}")