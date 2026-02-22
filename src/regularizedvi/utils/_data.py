"""Data download utilities."""

import os
import subprocess


def download_bone_marrow_dataset(data_folder: str = "data/") -> str:
    """Download the NeurIPS 2021 bone marrow multiome dataset.

    Downloads the curated h5ad file from the Open Problems S3 bucket if not
    already present locally.

    Parameters
    ----------
    data_folder
        Local directory for storing the downloaded file.

    Returns
    -------
    Path to the downloaded h5ad file.
    """
    os.makedirs(data_folder, exist_ok=True)

    h5ad_name = "bmmc_multiome_multivi_neurips21_curated.h5ad"
    h5ad_path = os.path.join(data_folder, h5ad_name)

    if not os.path.exists(h5ad_path):
        s3_uri = f"s3://openproblems-bio/public/post_competition/multiome/{h5ad_name}"
        print(f"Downloading {h5ad_name} from Open Problems S3 bucket...")
        subprocess.run(
            ["aws", "s3", "cp", s3_uri, h5ad_path, "--no-sign-request"],
            check=True,
        )
        print("Download complete.")
    else:
        print(f"Found {h5ad_path}")

    return h5ad_path
