#!/usr/bin/env python3
import numpy as np
import argparse
from scipy.ndimage import gaussian_filter

# Density table from tablas-positrones/input-materiales.dat.
# Index 0 is unused so material IDs can be used directly as indices.
MATERIAL_DENSITIES = np.array([
    0.0,
    1.21000000E-03,
    4.80000000E-01,
    9.26455000E-01,
    9.57722500E-01,
    9.84512500E-01,
    1.01130250E+00,
    1.02960900E+00,
    1.06086550E+00,
    1.12000000E+00,
    1.11172000E+00,
    1.16500000E+00,
    1.22420000E+00,
    1.28340000E+00,
    1.34260000E+00,
    1.40180000E+00,
    1.46100000E+00,
    1.52020000E+00,
    1.57940000E+00,
    1.63860000E+00,
    1.69780000E+00,
    1.75700000E+00,
    1.81620000E+00,
    1.87540000E+00,
    1.93460000E+00
], dtype=np.float32)


def density_from_material(material):
    """Map material IDs to densities using the fixed material table."""
    return MATERIAL_DENSITIES[material]


def real_density_from_ct(data):
    """Compute voxel density from CT (HU) values using piecewise calibration."""
    density = np.full(data.shape, 1.21e-3, dtype=np.float32)

    mask1 = data <= -950
    density[mask1] = 1.21e-3

    mask2 = (data > -950) & (data <= -120)
    density[mask2] = 1.031 + 0.00103 * data[mask2]

    mask3 = (data > -120) & (data <= -83)
    mask3a = mask3 & (data <= -98)
    mask3b = mask3 & (data > -98)
    density[mask3a] = 1.031 + 0.00103 * data[mask3a]
    density[mask3b] = 1.018 + 0.893e-3 * data[mask3b]

    mask4_7 = (data > -83) & (data <= 18)
    density[mask4_7] = 1.018 + 0.893e-3 * data[mask4_7]

    mask8_9 = (data > 18) & (data <= 120)
    density[mask8_9] = 1.003 + 1.169e-3 * data[mask8_9]

    mask10_24 = data > 120
    density[mask10_24] = 1.017 + 0.592e-3 * data[mask10_24]

    return density


def process_ct(data):
    """Process CT data and return material IDs."""
    material = np.ones(data.shape, dtype=np.int32)

    # Material segmentation based on CT values
    conditions = [
        data <= -950,
        (data > -950) & (data <= -120),
        (data > -120) & (data <= -83),
        (data > -83) & (data <= -53),
        (data > -53) & (data <= -23),
        (data > -23) & (data <= 7),
        (data > 7) & (data <= 18),
        (data > 18) & (data <= 80),
        (data > 80) & (data <= 120),
        (data > 120) & (data <= 200),
        (data > 200) & (data <= 300),
        (data > 300) & (data <= 400),
        (data > 400) & (data <= 500),
        (data > 500) & (data <= 600),
        (data > 600) & (data <= 700),
        (data > 700) & (data <= 800),
        (data > 800) & (data <= 900),
        (data > 900) & (data <= 1000),
        (data > 1000) & (data <= 1100),
        (data > 1100) & (data <= 1200),
        (data > 1200) & (data <= 1300),
        (data > 1300) & (data <= 1400),
        (data > 1400) & (data <= 1500),
        data > 1500
    ]

    # Assign material IDs
    for i, condition in enumerate(conditions):
        material[condition] = i + 1

    return material

def process_umap(data):
    """Process UMAP data: convert to CT, then to material IDs."""
    # Convert UMAP to CT
    ct_data = np.where(
        data <= 0.093,
        -1000.0 + 1000.0 * (data - 0.000111) / (0.093 - 0.000111),
        2000.0 * (data - 0.093) / (0.13 - 0.093)
    )

    return process_ct(ct_data)


def real_density_from_umap(data):
    """Compute real density from UMAP attenuation values."""
    att_511_factor = 0.096  # cm-1
    return data / att_511_factor

def read_binary_file(filename):
    """Read binary file as float32 array."""
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data

def write_binary_file(filename, data, dtype):
    """Write data to binary file."""
    with open(filename, 'wb') as f:
        f.write(data.astype(dtype).tobytes())

def reshape_flat_volume(flat, shape):
    shape = tuple(int(s) for s in shape)
    if len(shape) != 3:
        raise ValueError(f"Shape must have 3 dimensions, got {shape}.")
    expected = int(np.prod(shape))
    if flat.size != expected:
        raise ValueError(
            f"Input has {flat.size} voxels but shape {shape} implies {expected}."
        )
    return flat.reshape(shape, order="C")


def flatten_volume(volume_xyz):
    return np.ascontiguousarray(volume_xyz).reshape(-1)


def blur_density_correction(density_correction, shape, sigma):
    volume = reshape_flat_volume(density_correction, shape)
    blurred = gaussian_filter(volume, sigma=float(sigma), mode='reflect')
    return flatten_volume(blurred.astype(np.float32, copy=False))


def main():
    parser = argparse.ArgumentParser(description='Process CT or UMAP data to generate MATERIAL.raw and DENSITY.raw')
    parser.add_argument('input_file', help='Input binary file (CT or UMAP)')
    parser.add_argument('--type', choices=['ct', 'umap'], required=True, help='Input data type')
    parser.add_argument('--shape', nargs=3, type=int, default=[440, 440, 159], metavar=('X', 'Y', 'Z'),
                        help='Logical volume shape for the input RAW data. Default: 440 440 159')
    parser.add_argument('--blur-sigma', type=float, default=1.0,
                        help='Isotropic Gaussian sigma for DENSCORR blur. Default: 1.0')

    args = parser.parse_args()

    # Read input data
    data = read_binary_file(args.input_file)
    shape = tuple(args.shape)
    blur_sigma = float(args.blur_sigma)

    # Select processor by input type and run it
    processors = {
        'ct': (process_ct, real_density_from_ct),
        'umap': (process_umap, real_density_from_umap)
    }
    material = processors[args.type][0](data)
    density = density_from_material(material)
    real_density = processors[args.type][1](data)

    mask_correction = density > density.min()
    density_correction = real_density / density
    density_correction[~mask_correction] = 1.0
    print(f"Density correction factor: \
          min={density_correction.min():.3f}, \
          max={density_correction.max():.3f}, \
          mean={density_correction.mean():.3f}")   

    # Write output files
    write_binary_file('MATERIAL.raw', material, np.int32)
    write_binary_file('DENSITY.raw', density, np.float32)
    write_binary_file('DENSCORR.raw', density_correction, np.float32)

    density_correction_blur = blur_density_correction(
        density_correction, shape, blur_sigma
    )
    write_binary_file('DENSCORR.blur.raw', density_correction_blur, np.float32)

    print(f"Processed {len(data)} voxels")
    print(f"Generated MATERIAL.raw, DENSITY.raw, DENSCORR.raw and DENSCORR.blur.raw")
    print(f"Blur parameters: sigma={blur_sigma}, shape={shape}")

if __name__ == '__main__':
    main()
