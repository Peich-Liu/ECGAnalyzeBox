import numpy as np
import matplotlib.pyplot as plt

def calculate_sd1_sd2(rr_intervals):
    # Calculate mean of RR intervals
    rr_mean = np.mean(rr_intervals)

    # Calculate RR interval pairs (RR_n, RR_n+1)
    rr_n = rr_intervals[:-1]
    rr_n1 = rr_intervals[1:]

    # Calculate SD1 and SD2
    diff_rr = rr_n1 - rr_n
    sd1 = np.sqrt(np.var(diff_rr) / 2)
    sd2 = np.sqrt(2 * np.var(rr_intervals) - sd1 ** 2)

    return sd1, sd2, rr_mean

def plot_poincare(rr_intervals, sd1, sd2, rr_mean):
    rr_n = rr_intervals[:-1]
    rr_n1 = rr_intervals[1:]

    plt.figure(figsize=(8, 8))
    plt.scatter(rr_n, rr_n1, c='b', alpha=0.6, label='RR Intervals')

    # Plot average RR line
    plt.axvline(x=rr_mean, color='g', linestyle='--', label='Avg R-R interval')
    plt.axhline(y=rr_mean, color='g', linestyle='--')

    # Plot SD1 and SD2 ellipse
    angle = 45  # The angle for SD1 and SD2
    ellipse = plt.matplotlib.patches.Ellipse((rr_mean, rr_mean), 2*sd2, 2*sd1, angle=angle,
                                             edgecolor='r', facecolor='none', linestyle='-', linewidth=2, label='SD1/SD2 Ellipse')
    plt.gca().add_patch(ellipse)

    plt.xlabel('RR(n) (seconds)')
    plt.ylabel('RR(n+1) (seconds)')
    plt.title('Poincare Plot')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# Example usage with RR intervals in seconds
rr_intervals = np.array([0.8, 0.81, 0.815, 0.79, 0.805, 0.82, 0.83, 0.81, 0.79, 0.8])

# Calculate SD1, SD2, and mean RR interval
sd1, sd2, rr_mean = calculate_sd1_sd2(rr_intervals)
print(f'SD1: {sd1:.2f}, SD2: {sd2:.2f}')

# Plot Poincare Plot with SD1, SD2, and mean RR line
plot_poincare(rr_intervals, sd1, sd2, rr_mean)
