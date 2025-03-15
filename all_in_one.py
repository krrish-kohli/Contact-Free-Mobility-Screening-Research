import matplotlib.pyplot as plt
import numpy as np
from static_distance_gui import measure_distance
from range_velocity import measure_velocity
from new_angle import measure_angle


def acquire_raw_data():
    """
    Replace this function with your actual data acquisition logic.
    It should return the raw data (or a dictionary containing separate data
    for range/velocity and angle) that your measurement functions require.
    """
    # For demonstration, here we simulate raw data.
    num_chirps = 128
    num_samples = 256
    num_rx_antennas = 4

    # Simulated data for range/velocity (2D array)
    t = np.linspace(0, 1, num_samples)
    raw_data_range = np.array([np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.randn(num_samples)
                               for _ in range(num_chirps)])

    # Simulated data for angle (3D array, with extra antenna dimension)
    raw_data_angle = np.random.randn(num_chirps, num_samples, num_rx_antennas)

    # Depending on your implementation, you might return a dictionary or two separate arrays.
    return {
        'range': raw_data_range,
        'angle': raw_data_angle,
        'num_chirps': num_chirps,
        'num_samples': num_samples,
        'num_rx_antennas': num_rx_antennas
    }


def main():
    # Acquire a single set of raw data.
    raw_data = acquire_raw_data()

    # Process the same raw data for all measurements.
    # These functions are assumed to be designed to process the input data and return
    # either measurement values or the data to be plotted.
    distance_result = measure_distance(raw_data['range'])
    velocity_result = measure_velocity(raw_data['range'])
    angle_result = measure_angle(raw_data['angle'])

    # For demonstration, assume each function returns a dictionary like:
    # {'value': <measured value>, 'profile': <array for plotting>}
    # Adjust this based on your actual function output.

    # Create a figure with subplots to display all results together.
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    axs[0].plot(distance_result['profile'])
    axs[0].set_title(f"Distance Profile (Peak: {distance_result['value']:.2f} m)")
    axs[0].set_xlabel("Range bins")
    axs[0].set_ylabel("Amplitude (dB)")

    axs[1].plot(velocity_result['profile'])
    axs[1].set_title(f"Velocity Profile (Value: {velocity_result['value']})")
    axs[1].set_xlabel("Doppler bins")
    axs[1].set_ylabel("Amplitude (dB)")

    axs[2].plot(angle_result['profile'])
    axs[2].set_title(f"Angle Profile (Peak Angle: {angle_result['value']:.2f}°)")
    axs[2].set_xlabel("Angle bins")
    axs[2].set_ylabel("Amplitude (dB)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
