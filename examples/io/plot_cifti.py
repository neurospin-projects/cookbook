"""
Load CIFTI files
================

Utilities to load and decompose CIFTI-2 files.

The main entry point is `decompose_cifti()`, which returns:
- a 3D or 4D NIfTI volume (or raw masked volume array),
- a left-hemisphere surface data array,
- a right-hemisphere surface data array.

These functions rely on NiBabel's CIFTI-2 API.
"""

import numpy as np
import nibabel
import requests


# -------------------------------------------------------------------------
# Main decomposition function
# -------------------------------------------------------------------------

def decompose_cifti(cifti_file, raw=False):
    """
    Decompose a CIFTI-2 file into volume, left-surface, and right-surface data.

    Parameters
    ----------
    cifti_file : str
        Path to a CIFTI-2 image (.nii or .nii.gz).
    raw : bool, default False
        If True, return raw data arrays exactly as stored in the CIFTI file.
        If False, return reconstructed full-resolution arrays:
        - volume as a NIfTI image,
        - surfaces as full vertex arrays.

    Returns
    -------
    vol : array or nibabel.Nifti1Image
        Volume data extracted from the CIFTI file. If raw=False, this is a
        NIfTI image with the correct shape and affine.
    surf_left : array
        Left-hemisphere surface data, either raw or reconstructed to full
        vertex space.
    surf_right : array
        Right-hemisphere surface data, either raw or reconstructed to full
        vertex space.

    Notes
    -----
    A CIFTI file may contain multiple axes. This function identifies the
    BrainModelAxis, which encodes how data map to brain structures (volume
    voxels, cortical vertices, subcortical structures).
    """
    img = nibabel.load(cifti_file)
    data = img.get_fdata(dtype=np.float32)
    hdr = img.header

    axes = [hdr.get_axis(idx) for idx in range(img.ndim)]
    select_axes = [
        axis for axis in axes
        if isinstance(axis, nibabel.cifti2.BrainModelAxis)
    ]
    assert len(select_axes) == 1, (
        "Expected exactly one BrainModelAxis in CIFTI file."
    )
    brain_models = select_axes[0]


    vol = volume_from_cifti(data, brain_models, raw)
    left = surf_data_from_cifti(
        data, brain_models, "CIFTI_STRUCTURE_CORTEX_LEFT", raw
    )
    right = surf_data_from_cifti(
        data, brain_models, "CIFTI_STRUCTURE_CORTEX_RIGHT", raw
    )

    return vol, left, right


# -------------------------------------------------------------------------
# Surface extraction
# -------------------------------------------------------------------------

def surf_data_from_cifti(data, axis, surf_name, raw=False):
    """
    Extract surface data for a given cortical structure from a CIFTI file.

    Parameters
    ----------
    data : array
        Full CIFTI data array (time × brain models or vice versa).
    axis : nibabel.cifti2.BrainModelAxis
        The BrainModelAxis describing how data map to brain structures.
    surf_name : str
        Name of the cortical structure to extract, e.g.
        "CIFTI_STRUCTURE_CORTEX_LEFT" or "CIFTI_STRUCTURE_CORTEX_RIGHT".
    raw : bool, default False
        If True, return only the raw data rows corresponding to the structure.
        If False, return a full vertex array with missing vertices filled with 0.

    Returns
    -------
    surf_data : array
        Surface data for the requested structure.

    Raises
    ------
    ValueError
        If the requested structure is not present in the CIFTI file.

    Notes
    -----
    See https://nbviewer.org/github/neurohackademy/nh2020-curriculum/blob/
    master/we-nibabel-markiewicz/NiBabel.ipynb
    """
    assert isinstance(axis, nibabel.cifti2.BrainModelAxis)

    for name, data_indices, model in axis.iter_structures():
        if name == surf_name:
            data = data.T[data_indices]
            if raw:
                return data
            vtx_indices = model.vertex
            surf_data = np.zeros(
                (vtx_indices.max() + 1,) + data.shape[1:], dtype=data.dtype
            )
            surf_data[vtx_indices] = data
            return surf_data

    raise ValueError(f"No structure named {surf_name} found in CIFTI file.")


# -------------------------------------------------------------------------
# Volume extraction
# -------------------------------------------------------------------------

def volume_from_cifti(data, axis, raw=False):
    """
    Extract volume data from a CIFTI file.

    Parameters
    ----------
    data : array
        Full CIFTI data array.
    axis : nibabel.cifti2.BrainModelAxis
        The BrainModelAxis describing how data map to brain structures.
    raw : bool, default False
        If True, return only the raw masked volume rows.
        If False, reconstruct a full 3D or 4D NIfTI volume.

    Returns
    -------
    vol : array or nibabel.Nifti1Image
        Volume data extracted from the CIFTI file.

    Notes
    -----
    See https://nbviewer.org/github/neurohackademy/nh2020-curriculum/blob/
    master/we-nibabel-markiewicz/NiBabel.ipynb
    """
    assert isinstance(axis, nibabel.cifti2.BrainModelAxis)

    data = data.T[axis.volume_mask]

    if raw:
        return data

    vox_indices = tuple(axis.voxel[axis.volume_mask].T)
    vol_data = np.zeros(
        axis.volume_shape + data.shape[1:], dtype=data.dtype
    )
    vol_data[vox_indices] = data

    return nibabel.Nifti1Image(vol_data, axis.affine)


# Public URL containing a CIFTI file
url = (
    "https://raw.githubusercontent.com/GrattonLab/SeitzmanGratton-2019-PNAS/"
    "master/WashU120_groupNetworks.dtseries.nii"
)
cifti_path = "/tmp/sample.dtseries.nii"
print("Downloading...")
response = requests.get(url)
response.raise_for_status()
with open(cifti_path, "wb") as of:
    of.write(response.content)
print(f"Saved: {cifti_path}")

# Load and decompose the CIFTI file
vol, surf_left, surf_right = decompose_cifti(cifti_path)

# Print information about the extracted data
print("CIFTI decomposition summary:")
print(f"- volume: {vol.shape}")
print(f"- left cortex surface:  {surf_left.shape}")
print(f"- right cortex surface: {surf_right.shape}")
