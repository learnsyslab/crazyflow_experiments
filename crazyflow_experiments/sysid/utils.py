"""Sysid utils."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


def csv2dict(filepath: Path) -> dict[str, NDArray]:
    """Converts a csv into a dict of arrays."""
    rawdata = np.genfromtxt(filepath, delimiter=",", names=True)
    rawdict = {name: rawdata[name] for name in rawdata.dtype.names}

    # Following is chatgpt code
    data_dict = {}

    for key in rawdict:
        match = re.match(r"(\D+)(\d+)$", key)  # matches names like 'pos0', 'acc1', etc.
        if match:
            base, idx = match.group(1), int(match.group(2))
            if base not in data_dict:
                data_dict[base] = {}
            data_dict[base][idx] = rawdict[key]
        else:
            data_dict[key] = rawdict[key]

    # Convert data_dict sub-dicts into 2D arrays
    for key in data_dict:
        if isinstance(data_dict[key], dict):
            # Sort by index and stack columns
            cols = [data_dict[key][i] for i in sorted(data_dict[key])]
            data_dict[key] = np.stack(cols, axis=-1)

    return data_dict


def dict2csv(data: dict, filepath: Path):
    """Write one- or two-dimensional data arrays to a CSV file."""
    header = []
    rawdata = []
    for key, value in data.items():
        if key != "dt":
            if value.ndim == 1:
                header.append(key)
                rawdata.append(value)
            elif value.ndim == 2:
                for i in range(value.shape[1]):
                    header.append(f"{key}{i}")
                    rawdata.append(value[:, i])
            else:
                raise ValueError("Data must have at most 2 dim!")

    header = ",".join(header)
    rawdata = np.array(rawdata).T

    np.savetxt(filepath, rawdata, delimiter=",", header=header, comments="")
