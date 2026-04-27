import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define the paths to the plots
plots = [
    ("Confusion Matrix", "confusion_matrix.png"),
    ("Accuracy vs SNR", "accuracy_vs_snr.png"),
    ("Training Loss", "training_loss.png")
]

# Set up the figure for displaying the results
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('SDR Modulation Scheme Classification Results', fontsize=20, fontweight='bold', y=1.05)

for ax, (title, filename) in zip(axes, plots):
    if os.path.exists(filename):
        img = mpimg.imread(filename)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(title, fontsize=16)
    else:
        ax.text(0.5, 0.5, f"{filename} not found", ha='center', va='center')
        ax.axis('off')

plt.tight_layout()

# Print a nice summary to the console
print("="*60)
print(" MODULATION CLASSIFICATION MODEL RESULTS ")
print("="*60)
print("Model Architecture: 1D Convolutional Neural Network (CNN)")
print("Total Classes: 12 (4 Analog, 8 Digital)")
print("Data Shape: 1024 IQ Samples per segment")
print("\nEvaluation metrics have been saved as images in your folder:")
print(" 1. confusion_matrix.png - Shows classification performance per scheme")
print(" 2. accuracy_vs_snr.png - Displays robustness against simulated AWGN noise (-10dB to 20dB)")
print(" 3. training_loss.png - Shows model convergence over 30 epochs")
print("\nPopping up the visual dashboard for the professor...")
print("="*60)

plt.show()
