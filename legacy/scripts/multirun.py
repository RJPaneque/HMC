#!/usr/bin/env python3

import os
import subprocess
import shutil
import sys

# Define isotopes to process
ISOTOPES = [
    "Ga68",
    "I124",
    "Rb82",
]

# Parameters
SIZE = [121, 121, 121] # voxels
STEP = [0.1, 0.1, 0.1] # cm

def run_simulation(isotope, size, step, device=1):
    """Run simulation for a given isotope."""
    # Copy spectrum file
    src_spectrum = f"spectra/{isotope}.txt"
    dest_spectrum = "spectrum.txt"
    
    if not os.path.exists(src_spectrum):
        print(f"Warning: {src_spectrum} not found, skipping {isotope}")
        return False
    
    try:
        shutil.copy(src_spectrum, dest_spectrum)
        print(f"Copied {src_spectrum} to {dest_spectrum}")
    except Exception as e:
        print(f"Error copying spectrum for {isotope}: {e}")
        return False
    
    # Run the simulation
    cmd = f"./hybrid_positrons_UCM.x {device} {' '.join(map(str, size))} {' '.join(map(str, step))}"
    print(f"Running: {cmd} for isotope {isotope}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print(f"Successfully completed simulation for {isotope}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running simulation for {isotope}: {e}\n")
        return False

def main():
    if not ISOTOPES:
        print("Error: No isotopes defined in ISOTOPES list")
        sys.exit(1)
    
    successful = 0
    failed = 0
    
    for isotope in ISOTOPES:
        if run_simulation(isotope, SIZE, STEP):
            successful += 1
        else:
            failed += 1
    
    print(f"\nSimulation Summary:")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    main()
