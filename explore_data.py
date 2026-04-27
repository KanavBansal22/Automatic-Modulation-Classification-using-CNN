import os
import glob
import numpy as np

data_dir = r"c:\Users\lenovo\Downloads\comms proj\comms proj"
files = os.listdir(data_dir)

print(f"Total files: {len(files)}")
modulations = set()
for f in files:
    # strip digits to find modulation name
    name = "".join([c for c in f if not c.isdigit()])
    modulations.add(name)

print("Detected modulations:", list(modulations))

# Read one file
sample_file = os.path.join(data_dir, files[0])
size = os.path.getsize(sample_file)
print(f"File size: {size} bytes")
if size == 32768:
    print("Likely 4096 complex64 samples (4096 * 8 bytes = 32768)")

try:
    data = np.fromfile(sample_file, dtype=np.complex64)
    print("Data shape if complex64:", data.shape)
    print("First 5 samples:", data[:5])
except Exception as e:
    print("Error reading as complex64:", e)
