import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy import stats

def plot_channel_means_normalized_with_identity_r2(data, channel):
    """
    Compute the mean over time for each sample in a given channel, z-score normalize these means,
    and then plot both a histogram and a Q-Q plot against the standard normal distribution.
    In addition, compute the R² value for the Q-Q plot with respect to the identity line,
    i.e. how well the observed quantiles (sorted z-values) match the theoretical quantiles.
    
    Args:
        data: numpy.ndarray of shape (num_samples, num_channels, num_timepoints)
        channel: int, the channel index to analyze.
    
    Returns:
        z_means: 1D numpy array of z-score normalized means.
        id_r2: The R² value for the Q-Q plot when comparing the observed quantiles directly to the theoretical quantiles.
    """
    num_channels = data.shape[1]
    if channel < 0 or channel >= num_channels:
        raise ValueError(f"Channel index must be between 0 and {num_channels - 1}")
    
    # Compute means over the time axis for the specified channel.
    means = np.mean(data[:, channel, :], axis=1)
    
    # Z-score normalize the means.
    overall_mean = np.mean(means)
    overall_std = np.std(means, ddof=1) + 1e-8
    z_means = (means - overall_mean) / overall_std

    # Obtain theoretical quantiles and ordered observed quantiles without fitting.
    osm, osr = stats.probplot(z_means, dist="norm", fit=False)
    
    # Compute R² for the identity function: predicted = theoretical quantiles (osm)
    # Here SSE is the sum of squared errors between the observed quantiles (osr) and the theoretical ones (osm)
    # and SST is the total sum of squares of osr.
    sse = np.sum((osr - osm)**2)
    sst = np.sum((osr - np.mean(osr))**2)
    id_r2 = 1 - sse / sst

    # Create subplots: left for histogram, right for Q-Q plot.
    fig, (ax_hist, ax_qq) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram plot.
    ax_hist.hist(z_means, bins=30, color='skyblue', edgecolor='black')
    ax_hist.set_xlabel("Z-score of Mean Values", fontsize=14)
    ax_hist.set_ylabel("Count", fontsize=14)
    ax_hist.set_title(f"Histogram of Z-Normalized Means (Channel {channel})", fontsize=16)
    ax_hist.grid(True)
    
    # Q-Q Plot.
    stats.probplot(z_means, dist="norm", plot=ax_qq)
    ax_qq.set_title(f"Q-Q Plot (Channel {channel})\nIdentity R² = {id_r2:.3f}", fontsize=16)
    ax_qq.grid(True)

    plt.tight_layout()
    plt.show()
    
    return z_means, id_r2

def plot_all_channels_identity_r2(data):
    """
    Compute the identity R² (against the theoretical quantiles) for each channel
    and plot these R² values as a bar chart.
    
    Args:
        data: numpy.ndarray of shape (num_samples, num_channels, num_timepoints)
    
    Returns:
        r2_values: numpy array with one R² value for each channel.
    """
    num_channels = data.shape[1]
    r2_values = []
    for ch in range(num_channels):
        means = np.mean(data[:, ch, :], axis=1)
        z_means = (means - np.mean(means)) / (np.std(means, ddof=1) + 1e-8)
        osm, osr = stats.probplot(z_means, dist="norm", fit=False)
        sse = np.sum((osr - osm)**2)
        sst = np.sum((osr - np.mean(osr))**2)
        r2 = 1 - sse / sst
        r2_values.append(r2)
    r2_values = np.array(r2_values)
    
    # Bar plot of identity R² across channels.
    plt.figure(figsize=(12, 6))
    channels = np.arange(num_channels)
    plt.bar(channels, r2_values, color='lightgreen', edgecolor='black')
    plt.xlabel("Channel Index", fontsize=14)
    plt.ylabel("Identity R²", fontsize=14)
    plt.title("Identity R² of Q-Q Plot for Each Channel", fontsize=16)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return r2_values

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot histogram & Q-Q plot for a specified channel and compute identity R² for each channel.")
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to the numpy file (.npy) containing the EEG data "
                             "with shape (num_samples, num_channels, num_timepoints).")
    parser.add_argument("--channel", type=int, default=None,
                        help="Channel index to plot (0-indexed). If not provided, only the bar plot for all channels is shown.")
    args = parser.parse_args()

    # Load EEG data
    data = np.load(args.data_file)
    print("Data shape:", data.shape)
    
    # Plot bar plot for identity R² for all channels.
    all_r2 = plot_all_channels_identity_r2(data)
    
    # If a specific channel is provided, plot its histogram and Q-Q plot.
    if args.channel is not None:
        z_means, id_r2 = plot_channel_means_normalized_with_identity_r2(data, args.channel)
        print(f"Channel {args.channel}: Identity R² = {id_r2:.3f}")
