import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
import time
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
from scipy import constants

from ifxradarsdk import get_version_full
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwMetrics, FmcwSequenceChirp
from ifxradarsdk.common.exceptions import ErrorFrameAcquisitionFailed


class RadarProcessor:
    """Class to handle radar sensor configuration and data processing"""

    def __init__(self, config=None, data_dir=None):
        """Initialize the radar processing system

        Args:
            config: Optional radar configuration parameters
            data_dir: Directory for saving data, if None uses default
        """
        # Set default configuration if not provided
        self.config = config if config else self._get_default_config()

        # Data directory setup
        self.data_dir = data_dir if data_dir else os.path.join(os.path.expanduser("~"), "radar_data")
        os.makedirs(self.data_dir, exist_ok=True)

        # Initialize device
        self.device = DeviceFmcw()
        print(f"Radar SDK Version: {get_version_full()}")
        print(f"Sensor: {self.device.get_sensor_type()}")

        # Get number of RX antennas and initialize the device
        self.num_rx_antennas = self.device.get_sensor_information()["num_rx_antennas"]
        print(f"Number of RX antennas: {self.num_rx_antennas}")

        # Configure the device
        self._configure_device()

        # Create processing algorithms
        self._initialize_algorithms()

        # Data storage
        self.frame_index = 0
        self.timestamps = []
        self.distances = []
        self.velocities = []
        self.angles = []

        # Start timing
        self.start_time = time.time()

    def _get_default_config(self):
        """Get default radar configuration"""
        return {
            'frame_rate': 5,  # Hz - reduced from 10 to avoid acquisition failures
            'num_chirps': 128,
            'num_samples': 256,
            'range_resolution_m': 0.05,
            'max_range_m': 5.0,
            'max_speed_m_s': 3.0,
            'speed_resolution_m_s': 0.1,
            'max_angle_degrees': 60,
            'num_beams': 31,  # should be odd number for centered beam
        }

    def _configure_device(self):
        """Configure the radar device for our requirements"""
        # Create metrics for the device
        metrics = FmcwMetrics(
            range_resolution_m=self.config['range_resolution_m'],
            max_range_m=self.config['max_range_m'],
            max_speed_m_s=self.config['max_speed_m_s'],
            speed_resolution_m_s=self.config['speed_resolution_m_s'],
            center_frequency_Hz=60_750_000_000,  # 60.75 GHz - typical for these sensors
        )

        # Create acquisition sequence based on metrics parameters
        sequence = self.device.create_simple_sequence(FmcwSimpleSequenceConfig())
        sequence.loop.repetition_time_s = 1 / self.config['frame_rate']  # Set frame repetition time

        # Convert metrics into chirp loop parameters
        chirp_loop = sequence.loop.sub_sequence.contents
        self.device.sequence_from_metrics(metrics, chirp_loop)

        # Set remaining chirp parameters which are not derived from metrics
        chirp = chirp_loop.loop.sub_sequence.contents.chirp
        chirp.sample_rate_Hz = 1_000_000  # 1 MHz
        chirp.rx_mask = (1 << self.num_rx_antennas) - 1  # Use all available RX antennas
        chirp.tx_mask = 1  # Use first TX antenna
        chirp.tx_power_level = 31  # Maximum power
        chirp.if_gain_dB = 33
        chirp.lp_cutoff_Hz = 500000  # Low-pass filter cutoff at 500 kHz
        chirp.hp_cutoff_Hz = 80000  # High-pass filter cutoff at 80 kHz

        # Update configuration with actual values
        self.config['num_chirps'] = chirp_loop.loop.num_repetitions
        self.config['num_samples'] = chirp.num_samples

        # Save the chirp and metrics for later reference
        self.chirp = chirp
        self.metrics = metrics

        # Apply configuration to device
        self.device.set_acquisition_sequence(sequence)

    def _initialize_algorithms(self):
        """Initialize the processing algorithms"""
        # Import algorithms here to avoid circular imports
        from helpers.DistanceAlgo import DistanceAlgo
        from helpers.DopplerAlgo import DopplerAlgo
        from helpers.DigitalBeamForming import DigitalBeamForming

        # Create processing algorithms
        self.range_algo = DistanceAlgo(self.chirp, self.config['num_chirps'])
        self.doppler_algo = DopplerAlgo(self.config['num_samples'], self.config['num_chirps'], self.num_rx_antennas)
        self.dbf = DigitalBeamForming(self.num_rx_antennas, self.config['num_beams'], self.config['max_angle_degrees'])

        # Initialize arrays for range-doppler spectrum
        self.rd_spectrum = np.zeros((self.config['num_samples'], 2 * self.config['num_chirps'], self.num_rx_antennas),
                                    dtype=complex)

    def process_frame(self):
        """Process a frame of radar data to extract distance, velocity, and angle"""
        # Get the next frame
        frame_contents = self.device.get_next_frame()
        frame_data = frame_contents[0]

        # Track time
        current_time = time.time() - self.start_time
        self.timestamps.append(current_time)

        # Initialize output containers
        distance_data_all_antennas = []
        distance_peaks = []
        velocity_data_all_antennas = []
        velocity_peaks = []

        # Calculate chirp duration and wavelength for Doppler resolution
        chirp_duration = 1.0 / (self.chirp.sample_rate_Hz / self.chirp.num_samples)
        wavelength = constants.c / self.metrics.center_frequency_Hz

        # Set the Doppler resolution if not already set
        if self.doppler_algo.doppler_resolution is None:
            self.doppler_algo.set_doppler_resolution(chirp_duration, wavelength)

        # Process range and doppler for each antenna
        for i_ant in range(self.num_rx_antennas):
            # Get antenna samples - shape: (num_chirps, num_samples)
            antenna_samples = frame_data[i_ant, :, :]

            # Process distance
            distance_peak_m, distance_data = self.range_algo.compute_distance(antenna_samples)
            distance_data_all_antennas.append(distance_data)
            distance_peaks.append(distance_peak_m)

            # Process velocity using the enhanced method
            velocity_m_s, rd_map = self.doppler_algo.compute_velocity(antenna_samples, i_ant)
            velocity_peaks.append(velocity_m_s)

            # Store range-doppler spectrum for beamforming
            self.rd_spectrum[:, :, i_ant] = rd_map

            # Convert to dB scale for visualization
            rd_map_db = 20 * np.log10(abs(rd_map))
            velocity_data_all_antennas.append(rd_map_db)

        # Use digital beamforming for angle estimation
        rd_beam_formed = self.dbf.run(self.rd_spectrum)

        # Calculate beam energy across range dimension
        beam_range_energy = np.zeros((self.config['num_samples'], self.config['num_beams']))
        for i_beam in range(self.config['num_beams']):
            doppler_i = rd_beam_formed[:, :, i_beam]
            beam_range_energy[:, i_beam] += np.linalg.norm(doppler_i, axis=1) / np.sqrt(self.config['num_beams'])

        # Find beam with maximum energy and determine angle
        max_energy_idx = np.unravel_index(beam_range_energy.argmax(), beam_range_energy.shape)
        angle_bin = max_energy_idx[1]
        angle_degrees = np.linspace(-self.config['max_angle_degrees'],
                                    self.config['max_angle_degrees'],
                                    self.config['num_beams'])[angle_bin]

        # Store results
        self.distances.append(np.mean(distance_peaks))
        self.velocities.append(np.mean(velocity_peaks))
        self.angles.append(angle_degrees)

        # Increment frame counter
        self.frame_index += 1

        # Return results
        results = {
            'time': current_time,
            'distance': np.mean(distance_peaks),
            'velocity': np.mean(velocity_peaks),
            'angle': angle_degrees,
            'distance_profiles': distance_data_all_antennas,
            'velocity_maps': velocity_data_all_antennas,
            'angle_profile': beam_range_energy,
            'frame_index': self.frame_index
        }

        return results

    def save_data(self, filename=None):
        """Save collected data to Excel file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"radar_data_{timestamp}.xlsx"

        file_path = os.path.join(self.data_dir, filename)

        # Create DataFrame
        df = pd.DataFrame({
            "Time (s)": self.timestamps,
            "Distance (m)": self.distances,
            "Velocity (m/s)": self.velocities,
            "Angle (degrees)": self.angles
        })

        # Save to Excel
        df.to_excel(file_path, index=False)
        print(f"Data saved to {file_path}")

        return file_path

    def close(self):
        """Clean up and close the device"""
        if hasattr(self, 'device'):
            del self.device


class RadarGUI:
    """GUI for radar processing and visualization"""

    def __init__(self, root):
        """Initialize the GUI

        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Radar Processing System")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Configuration variables
        self.config = None
        self.data_dir = os.path.join(os.path.expanduser("~"), "radar_data")
        self.processor = None
        self.processing_active = False
        self.end_time = None

        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create upper control frame
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Control Panel", padding="10")
        self.control_frame.pack(fill=tk.X, pady=5)

        # Duration input
        ttk.Label(self.control_frame, text="Capture Duration (seconds):").grid(row=0, column=0, padx=5, pady=5,
                                                                               sticky=tk.W)
        self.duration_var = tk.StringVar(value="30")
        self.duration_entry = ttk.Entry(self.control_frame, textvariable=self.duration_var, width=10)
        self.duration_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        # Directory selection
        ttk.Label(self.control_frame, text="Data Directory:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.dir_var = tk.StringVar(value=self.data_dir)
        ttk.Entry(self.control_frame, textvariable=self.dir_var, width=30).grid(row=0, column=3, padx=5, pady=5,
                                                                                sticky=tk.W)
        ttk.Button(self.control_frame, text="Browse...", command=self.browse_directory).grid(row=0, column=4, padx=5,
                                                                                             pady=5)

        # Control buttons
        self.start_button = ttk.Button(self.control_frame, text="Start Capture", command=self.start_capture)
        self.start_button.grid(row=0, column=5, padx=5, pady=5)

        self.stop_button = ttk.Button(self.control_frame, text="Stop Capture", command=self.stop_capture,
                                      state=tk.DISABLED)
        self.stop_button.grid(row=0, column=6, padx=5, pady=5)

        # Status frame
        self.status_frame = ttk.LabelFrame(self.main_frame, text="Status", padding="10")
        self.status_frame.pack(fill=tk.X, pady=5)

        # Status variables
        self.status_var = tk.StringVar(value="Ready")
        self.time_var = tk.StringVar(value="Time: 0.0 s")
        self.frames_var = tk.StringVar(value="Frames: 0")
        self.distance_var = tk.StringVar(value="Distance: 0.00 m")
        self.velocity_var = tk.StringVar(value="Velocity: 0.00 m/s")
        self.angle_var = tk.StringVar(value="Angle: 0.0°")

        # Status labels
        ttk.Label(self.status_frame, textvariable=self.status_var, font=("TkDefaultFont", 10, "bold")).grid(row=0,
                                                                                                            column=0,
                                                                                                            padx=5,
                                                                                                            pady=5,
                                                                                                            sticky=tk.W)
        ttk.Label(self.status_frame, textvariable=self.time_var).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Label(self.status_frame, textvariable=self.frames_var).grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        ttk.Label(self.status_frame, textvariable=self.distance_var).grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        ttk.Label(self.status_frame, textvariable=self.velocity_var).grid(row=0, column=4, padx=5, pady=5, sticky=tk.W)
        ttk.Label(self.status_frame, textvariable=self.angle_var).grid(row=0, column=5, padx=5, pady=5, sticky=tk.W)

        # Progress bar
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(self.status_frame, orient=tk.HORIZONTAL, length=200, mode='determinate',
                                            variable=self.progress_var)
        self.progress_bar.grid(row=0, column=6, padx=5, pady=5, sticky=tk.EW)

        # Plots frame
        self.plots_frame = ttk.LabelFrame(self.main_frame, text="Radar Data Visualization", padding="10")
        self.plots_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create figure for plots
        self.create_plots()

    def create_plots(self):
        """Create the matplotlib plots"""
        # Create figure with subplots
        self.fig = plt.Figure(figsize=(12, 8), dpi=100)

        # Create subplots
        gs = self.fig.add_gridspec(2, 3)

        # Distance profile
        self.ax_dist = self.fig.add_subplot(gs[0, 0])
        self.ax_dist.set_title("Distance Profile")
        self.ax_dist.set_xlabel("Distance (m)")
        self.ax_dist.set_ylabel("Magnitude (dB)")
        self.ax_dist.grid(True)

        # Velocity profile
        self.ax_vel = self.fig.add_subplot(gs[0, 1])
        self.ax_vel.set_title("Velocity Profile")
        self.ax_vel.set_xlabel("Velocity (m/s)")
        self.ax_vel.set_ylabel("Magnitude (dB)")
        self.ax_vel.grid(True)

        # Angle profile
        self.ax_angle = self.fig.add_subplot(gs[0, 2])
        self.ax_angle.set_title("Angle Profile")
        self.ax_angle.set_xlabel("Angle (°)")
        self.ax_angle.set_ylabel("Magnitude (dB)")
        self.ax_angle.grid(True)

        # Range-Doppler map
        self.ax_rd = self.fig.add_subplot(gs[1, :])
        self.ax_rd.set_title("Range-Doppler Map")
        self.ax_rd.set_xlabel("Velocity (m/s)")
        self.ax_rd.set_ylabel("Distance (m)")

        # Initial placeholder data
        self.dist_line, = self.ax_dist.plot([], [])
        self.vel_line, = self.ax_vel.plot([], [])
        self.angle_line, = self.ax_angle.plot([], [])

        # RD map placeholder
        self.rd_img = None

        # Add the plots to the tkinter interface
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plots_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add navigation toolbar
        toolbar_frame = ttk.Frame(self.plots_frame)
        toolbar_frame.pack(fill=tk.X)
        NavigationToolbar2Tk(self.canvas, toolbar_frame)

        # Adjust layout
        self.fig.tight_layout()

    def browse_directory(self):
        """Browse for a directory to save data"""
        directory = filedialog.askdirectory(initialdir=self.data_dir)
        if directory:
            self.data_dir = directory
            self.dir_var.set(directory)

    def start_capture(self):
        """Start the radar data capture"""
        try:
            # Get duration
            try:
                duration = float(self.duration_var.get())
                if duration <= 0:
                    raise ValueError("Duration must be positive")
            except ValueError as e:
                messagebox.showerror("Invalid Input", f"Invalid duration: {str(e)}")
                return

            # Update data directory
            self.data_dir = self.dir_var.get()
            os.makedirs(self.data_dir, exist_ok=True)

            # Initialize radar processor
            self.processor = RadarProcessor(data_dir=self.data_dir)

            # Set up state
            self.processing_active = True
            self.end_time = time.time() + duration

            # Update UI
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.duration_entry.config(state=tk.DISABLED)

            # Update status
            self.status_var.set("Capturing...")

            # Start processing
            self.root.after(10, self.process_frame)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start capture: {str(e)}")
            self.cleanup()

    def stop_capture(self):
        """Stop the radar data capture"""
        if self.processing_active:
            self.processing_active = False
            self.status_var.set("Stopping...")

    def process_frame(self):
        """Process a single radar frame"""
        if not self.processing_active:
            self.save_and_cleanup()
            return

        # Check for timeout
        current_time = time.time()
        if self.end_time and current_time >= self.end_time:
            self.processing_active = False
            self.save_and_cleanup()
            return

        try:
            # Process frame
            results = self.processor.process_frame()

            # Calculate progress
            if self.end_time:
                progress = min(100, (current_time - (self.end_time - float(self.duration_var.get()))) / float(
                    self.duration_var.get()) * 100)
                self.progress_var.set(progress)

            # Update status variables
            self.time_var.set(f"Time: {results['time']:.1f} s")
            self.frames_var.set(f"Frames: {results['frame_index']}")
            self.distance_var.set(f"Distance: {results['distance']:.2f} m")
            self.velocity_var.set(f"Velocity: {results['velocity']:.2f} m/s")
            self.angle_var.set(f"Angle: {results['angle']:.1f}°")

            # Update plots
            self.update_plots(results)

            # Schedule next frame
            self.root.after(10, self.process_frame)

        except ErrorFrameAcquisitionFailed:
            # Retry on frame acquisition failure
            print("Frame acquisition failed, retrying...")
            self.root.after(100, self.process_frame)

        except Exception as e:
            messagebox.showerror("Error", f"Processing error: {str(e)}")
            self.processing_active = False
            self.save_and_cleanup()

    def update_plots(self, results):
        """Update the plots with new data"""
        # Get data from results
        # Distance profile (first antenna)
        distance_profile = results['distance_profiles'][0]
        max_range = self.processor.config['max_range_m']
        range_axis = np.linspace(0, max_range, len(distance_profile))

        # Velocity profile (first antenna)
        velocity_map = results['velocity_maps'][0]
        velocity_profile = np.max(velocity_map, axis=0)
        max_speed = self.processor.config['max_speed_m_s']
        velocity_axis = np.linspace(-max_speed, max_speed, len(velocity_profile))

        # Angle profile
        angle_profile = np.max(results['angle_profile'], axis=0)
        max_angle = self.processor.config['max_angle_degrees']
        angle_axis = np.linspace(-max_angle, max_angle, len(angle_profile))

        # Update distance plot
        self.dist_line.set_data(range_axis, 20 * np.log10(distance_profile + 1e-10))
        self.ax_dist.set_xlim(0, max_range)
        self.ax_dist.set_ylim(np.min(20 * np.log10(distance_profile + 1e-10)) - 5,
                              np.max(20 * np.log10(distance_profile + 1e-10)) + 5)

        # Mark detected distance
        for line in self.ax_dist.get_lines()[1:]:
            line.remove()
        self.ax_dist.axvline(x=results['distance'], color='r', linestyle='--', alpha=0.7)

        # Update velocity plot
        self.vel_line.set_data(velocity_axis, velocity_profile)
        self.ax_vel.set_xlim(-max_speed, max_speed)
        self.ax_vel.set_ylim(np.min(velocity_profile) - 5, np.max(velocity_profile) + 5)

        # Mark detected velocity
        for line in self.ax_vel.get_lines()[1:]:
            line.remove()
        self.ax_vel.axvline(x=results['velocity'], color='r', linestyle='--', alpha=0.7)

        # Update angle plot
        self.angle_line.set_data(angle_axis, angle_profile)
        self.ax_angle.set_xlim(-max_angle, max_angle)
        self.ax_angle.set_ylim(np.min(angle_profile) - 5, np.max(angle_profile) + 5)

        # Mark detected angle
        for line in self.ax_angle.get_lines()[1:]:
            line.remove()
        self.ax_angle.axvline(x=results['angle'], color='r', linestyle='--', alpha=0.7)

        # Update range-Doppler map
        if self.rd_img is None:
            extent = [-max_speed, max_speed, 0, max_range]
            self.rd_img = self.ax_rd.imshow(
                velocity_map,
                aspect='auto',
                origin='lower',
                cmap='viridis',
                extent=extent,
                interpolation='nearest'
            )
            self.fig.colorbar(self.rd_img, ax=self.ax_rd, label="Magnitude (dB)")
        else:
            self.rd_img.set_data(velocity_map)
            self.rd_img.set_clim(vmin=np.min(velocity_map), vmax=np.max(velocity_map))

        # Mark detected distance and velocity on RD map
        for line in self.ax_rd.get_lines():
            line.remove()
        self.ax_rd.axhline(y=results['distance'], color='r', linestyle='--', alpha=0.7)
        self.ax_rd.axvline(x=results['velocity'], color='r', linestyle='--', alpha=0.7)

        # Draw updated figure
        self.canvas.draw_idle()

    def save_and_cleanup(self):
        """Save data and clean up resources"""
        try:
            # Save data if processor exists and has frames
            if self.processor and self.processor.frame_index > 0:
                data_file = self.processor.save_data()
                messagebox.showinfo("Capture Complete",
                                    f"Radar data capture completed.\n\n"
                                    f"Processed {self.processor.frame_index} frames.\n"
                                    f"Data saved to:\n{data_file}")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving data: {str(e)}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources and reset UI"""
        # Close processor if it exists
        if self.processor:
            try:
                self.processor.close()
            except:
                pass
            self.processor = None

        # Reset UI elements
        self.processing_active = False
        self.end_time = None
        self.progress_var.set(0)
        self.status_var.set("Ready")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.duration_entry.config(state=tk.NORMAL)

    def on_closing(self):
        """Handle window closing event"""
        if self.processing_active:
            if messagebox.askokcancel("Quit", "Data capture is in progress. Do you want to stop and quit?"):
                self.stop_capture()
                self.root.after(1000, self.root.destroy)
        else:
            self.root.destroy()


def main():
    """Main function to run the radar processing GUI"""
    parser = argparse.ArgumentParser(description='Radar Processing System')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    args = parser.parse_args()

    # Load configuration if provided
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                import json
                config = json.load(f)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return

    # Create root window
    root = tk.Tk()
    app = RadarGUI(root)

    # Run the application
    root.mainloop()


if __name__ == "__main__":
    main()