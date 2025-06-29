
# Install required packages
# !pip install --quiet EMD-signal pandas matplotlib

import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD, EEMD
from google.colab import files

# Excel file upload
def upload_data():
    print("Please upload your Excel file (e.g., 'data.xlsx'):")
    uploaded = files.upload()
    excel_name = list(uploaded.keys())[0]
    df = pd.read_excel(io.BytesIO(uploaded[excel_name]))
    return df

# Compute and plot EMD
def plot_emd(data, t, title, df, exclude_high_freq=0):
    """
    exclude_high_freq: number of highest-frequency IMFs to exclude from combined reconstruction
    """
    emd = EMD()
    imfs = emd.emd(data, t)
    n_imfs = imfs.shape[0]
    fig, axes = plt.subplots(n_imfs + 2, 1, figsize=(12, 2*(n_imfs+2)))

    # Original signal
    axes[0].plot(t, data)
    axes[0].set_title(f'{title} - Original Signal (EMD)')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel(f'{title}')

    # Individual IMFs
    for i, imf in enumerate(imfs):
        axes[i+1].plot(t, imf)
        axes[i+1].set_title(f'{title} - EMD IMF {i+1}')
        axes[i+1].set_xlabel('Time (s)')
        axes[i+1].set_ylabel('Amplitude')

    # Combined reconstruction excluding highest-frequency IMFs
    exclude = min(exclude_high_freq, n_imfs)
    combined = np.sum(imfs[exclude:], axis=0)
    axes[-1].plot(t, combined)
    axes[-1].set_title(
        f'{title} - Combined Reconstruction (excluding first {exclude} IMF(s))'
    )
    axes[-1].set_xlabel('Time (s)')
    axes[-1].set_ylabel(f'{title} Recon')

    plt.tight_layout()
    plt.show()

    df[f"{title} Reconstructed"] = combined
    return imfs, combined

# Compute and plot EEMD
def plot_eemd(data, t, title, df, noise_width=0.05, trials=100, exclude_high_freq=0):
    """
    exclude_high_freq: number of highest-frequency IMFs to exclude from combined reconstruction
    """
    eemd = EEMD(trials=trials, noise_width=noise_width)
    imfs = eemd.eemd(data, t)
    n_imfs = imfs.shape[0]
    fig, axes = plt.subplots(n_imfs + 1, 1, figsize=(12, 2*(n_imfs+1)))

    # Individual IMFs
    for i, imf in enumerate(imfs):
        axes[i].plot(t, imf)
        axes[i].set_title(f'{title} - EEMD IMF {i+1}')
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('Amplitude')

    # Combined reconstruction excluding highest-frequency IMFs
    exclude = min(exclude_high_freq, n_imfs)
    combined = np.sum(imfs[exclude:], axis=0)
    axes[-1].plot(t, combined)
    axes[-1].set_title(
        f'{title} - Combined Reconstruction (EEMD, excluding first {exclude} IMF(s))'
    )
    axes[-1].set_xlabel('Time (s)')
    axes[-1].set_ylabel(f'{title} Recon')

    plt.tight_layout()
    plt.show()

    df[f"{title} Reconstructed"] = combined
    return imfs, combined

# Main execution
def main():
    df = upload_data()
    # Ask user for exclusion parameter
    exclude = int(input(
        'Enter number of highest-frequency IMFs to exclude for smoothing (0 to include all): '
    ))
    t = df.iloc[:, 0].values
    signals = {
        'Distance': df.iloc[:, 1].values,
        'Velocity': df.iloc[:, 2].values,
        'Angle':    df.iloc[:, 3].values
    }
    for name, sig in signals.items():
        print(f"\nProcessing {name} with exclude_high_freq={exclude}...")

        plot_emd(signals['Velocity'], t, 'Velocity', df=df, exclude_high_freq=exclude)
        plot_emd(signals['Angle'], t, 'Angle', df=df, exclude_high_freq=exclude)
        plot_emd(signals['Distance'], t, 'Distance', df=df, exclude_high_freq=exclude)

        plot_eemd(signals['Velocity'], t, 'Velocity', df=df, exclude_high_freq=exclude)
        plot_eemd(signals['Angle'], t, 'Angle', df=df, exclude_high_freq=exclude)
        plot_eemd(signals['Distance'], t, 'Distance', df=df, exclude_high_freq=exclude)

    # Save the DataFrame you uploaded (with standardized column names) so the round_detector can see it automatically:
    output_filename = 'emdandeemd_output.xlsx'
    df.to_excel(output_filename, index=False)
    print(f"✅ Data written to {output_filename}")

if __name__ == "__main__":
    main()