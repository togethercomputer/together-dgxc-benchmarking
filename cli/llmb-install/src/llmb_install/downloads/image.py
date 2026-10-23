# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


"""Container image management for LLMB installation.

This module provides functions for managing container images including image collection,
filename generation, and image fetching using enroot.
"""

import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from llmb_install.environment.detector import (
    detect_virtual_environment,
    get_clean_environment_for_subprocess,
    has_active_conda_environment,
)


# CVE-2026-111943: the vendor images pinned by the recipes ship a ca-certificates
# package that still trusts the compromised emSign CA. After `enroot import`, the
# image is unpacked, ca-certificates is upgraded from the distro security repo
# and the image is re-exported. The job fails if the package is still too old.
# Set LLMB_SKIP_CA_UPGRADE=1 to skip this (e.g. clusters without package-repo access).
CA_UPGRADE_MARKER_SUFFIX = ".ca-upgraded"

_CA_UPGRADE_IN_CONTAINER = r"""
set -e
if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get -o APT::Sandbox::User=root update
  apt-get -o APT::Sandbox::User=root install -y --only-upgrade ca-certificates
  rm -rf /var/lib/apt/lists/*
  . /etc/os-release
  case "$ID:$VERSION_ID" in
    ubuntu:*) min="20260816~" ;;
    debian:12) min="20250419~deb12u2" ;;
    debian:13) min="20250419+deb13u1" ;;
    *) min="" ;;
  esac
  ver=$(dpkg-query -W -f='${Version}' ca-certificates)
  if [ -n "$min" ] && ! dpkg --compare-versions "$ver" ge "$min"; then
    echo "ca-certificates $ver is older than $min (CVE-2026-111943)" >&2
    exit 1
  fi
elif command -v apk >/dev/null 2>&1; then
  apk upgrade --no-cache ca-certificates
  ver=$(apk info -e -v ca-certificates | sed 's/^ca-certificates-//')
  if [ "$(apk version -t "$ver" 20261020-r0)" = "<" ]; then
    echo "ca-certificates $ver is older than 20261020-r0 (CVE-2026-111943)" >&2
    exit 1
  fi
elif command -v dnf >/dev/null 2>&1; then
  dnf -y update ca-certificates && update-ca-trust
elif command -v microdnf >/dev/null 2>&1; then
  microdnf -y update ca-certificates && update-ca-trust
elif command -v yum >/dev/null 2>&1; then
  yum -y update ca-certificates && update-ca-trust
fi
"""

# Args: $1 output sqsh, $2 image URL ("" to upgrade an existing $1), $3 architecture,
# $4 in-container upgrade script, $5 "1" to skip the upgrade.
_IMPORT_AND_UPGRADE_SCRIPT = r"""
set -euo pipefail
OUT="$1"; URL="$2"; ARCH="$3"; UPGRADE="$4"; SKIP_UPGRADE="$5"
NAME="llmb-ca-upgrade-$(hostname -s)-$$"
SRC="$OUT"
cleanup() {
  enroot remove -f "$NAME" >/dev/null 2>&1 || true
  rm -f "$OUT.new.$$"
  if [ -n "$URL" ]; then rm -f "$OUT.raw.$$"; fi
}
trap cleanup EXIT
if [ -n "$URL" ]; then
  SRC="$OUT.raw.$$"
  if [ -n "$ARCH" ]; then
    enroot import -o "$SRC" -a "$ARCH" "docker://$URL"
  else
    enroot import -o "$SRC" "docker://$URL"
  fi
fi
if [ "$SKIP_UPGRADE" = "1" ]; then
  if [ "$SRC" != "$OUT" ]; then mv -f "$SRC" "$OUT"; fi
  exit 0
fi
enroot create -n "$NAME" "$SRC"
enroot start --root --rw "$NAME" /bin/sh -c "$UPGRADE"
enroot export -o "$OUT.new.$$" "$NAME"
mv -f "$OUT.new.$$" "$OUT"
"""


def is_enroot_installed() -> bool:
    """Check if Enroot is installed and available in the system PATH.

    Returns:
        bool: True if enroot command is available, False otherwise
    """
    return shutil.which('enroot') is not None


def get_required_images(workloads: Dict[str, Dict[str, Any]], selected_keys: List[str]) -> Dict[str, str]:
    """Get the dictionary of container images required by the selected workloads.

    Args:
        workloads: Dictionary of all available workloads
        selected_keys: List of selected workload keys

    Returns:
        Dict[str, str]: Dictionary mapping image URLs to their desired filenames
    """
    # Use filename as key to deduplicate images that have different URL formats
    # but represent the same image (e.g., nvcr.io/nvidia/nemo vs nvcr.io#nvidia/nemo)
    filename_to_url = {}

    for key in selected_keys:
        workload = workloads[key]
        container_config = workload.get('container', {})

        # Handle image field which can be a string, list, or list of dicts
        images = container_config.get('images', [])

        if isinstance(images, str):
            # Simple string format - generate filename from URL
            filename = _generate_image_filename(images)
            filename_to_url[filename] = images
        elif isinstance(images, list):
            for image in images:
                if isinstance(image, str):
                    # Simple string format - generate filename from URL
                    filename = _generate_image_filename(image)
                    filename_to_url[filename] = image
                elif isinstance(image, dict):
                    url = image.get('url')
                    name = image.get('name')
                    if url:
                        if name:
                            filename = f"{name}.sqsh"
                            filename_to_url[filename] = url
                        else:
                            filename = _generate_image_filename(url)
                            filename_to_url[filename] = url

    # Convert back to URL -> filename mapping for compatibility with existing code
    required_images = {url: filename for filename, url in filename_to_url.items()}
    return required_images


def _generate_image_filename(image_url: str) -> str:
    """Generate a filename for a container image URL.

    Args:
        image_url: Container image URL (e.g., 'nvcr.io/nvidia/nemo:25.02.01' or 'nvcr.io#nvidia/nemo:sha256:abc123...')

    Returns:
        str: Generated filename ending with .sqsh
    """
    try:
        # Normalize URL separators - replace # with / for consistent parsing
        normalized_url = image_url.replace('#', '/')

        # Extract parent name, image name and version/digest from image URL
        parts = normalized_url.split('/')
        parent = parts[-2]  # Get 'nvidia' from URL
        name_and_tag = parts[-1]  # Get 'nemo:25.02.01' or 'nemo:sha256:abc123...'

        if ':' in name_and_tag:
            name, tag_or_digest = name_and_tag.split(':', 1)

            # Check if this is a SHA digest
            if tag_or_digest.startswith('sha256:'):
                # For SHA digests, use first 12 characters of the hash
                hash_part = tag_or_digest.split(':', 1)[1][:12]
                version = f"sha256-{hash_part}"
            else:
                # Regular tag
                version = tag_or_digest
        else:
            # No tag specified, use 'latest'
            name = name_and_tag
            version = "latest"

        return f"{parent}+{name}+{version}.sqsh"

    except (IndexError, ValueError):
        # Fallback: use a sanitized version of the full URL
        sanitized = image_url.replace('/', '_').replace(':', '_').replace('#', '_')
        return f"{sanitized}.sqsh"


def fetch_container_images(
    images: Dict[str, str],
    install_path: str,
    node_architecture: str,
    install_method: str,
    slurm_info: Dict[str, Any],
    shared_images_dir: Optional[str] = None,
) -> None:
    """Fetch container images using enroot and save them as sqsh files.

    Args:
        images: Dictionary mapping image URLs to desired filenames
        install_path: Directory where the sqsh files should be saved
        node_architecture: The selected node architecture ('x86_64' or 'aarch64')
        install_method: The selected installation method ('local' or 'slurm')
        slurm_info: Dictionary containing SLURM configuration information
        shared_images_dir: Optional path to a shared directory for images.

    The function will:
    1. Use provided filenames or generate them from image URLs
    2. Create sqsh files using enroot import
    3. Skip if sqsh file already exists
    4. If a shared_images_dir is provided, store images there and symlink to install_path/images.
    """
    local_images_dir = os.path.join(install_path, "images")
    os.makedirs(local_images_dir, exist_ok=True)

    if shared_images_dir:
        source_images_dir = shared_images_dir
        print(f"\nUsing shared image location: {source_images_dir}")
        os.makedirs(source_images_dir, exist_ok=True)
    else:
        source_images_dir = local_images_dir

    # Detect virtual environment and prepare clean environment for SLURM
    if detect_virtual_environment() and install_method == 'slurm':
        print("Detected virtual environment. Preparing clean environment for SLURM execution...")
        clean_env = get_clean_environment_for_subprocess()
    else:
        clean_env = None

    for image_url, filename in sorted(images.items()):
        try:
            source_sqsh_file = os.path.join(source_images_dir, filename)

            ca_marker = source_sqsh_file + CA_UPGRADE_MARKER_SUFFIX
            skip_ca_upgrade = os.environ.get("LLMB_SKIP_CA_UPGRADE", "0") == "1"
            if os.path.exists(source_sqsh_file) and (skip_ca_upgrade or os.path.exists(ca_marker)):
                print(f"Skipping {image_url} -- {source_sqsh_file} - file already exists.")
            else:
                # Import (unless an older sqsh already exists) and upgrade ca-certificates in one job,
                # so both steps run on the same node.
                import_url = "" if os.path.exists(source_sqsh_file) else image_url
                cmd_args = [
                    "bash",
                    "-c",
                    _IMPORT_AND_UPGRADE_SCRIPT,
                    "llmb-image-import",
                    source_sqsh_file,
                    import_url,
                    node_architecture or "",
                    _CA_UPGRADE_IN_CONTAINER,
                    "1" if skip_ca_upgrade else "0",
                ]

                if install_method == 'slurm':
                    # Validate required SLURM configuration fields
                    try:
                        account = slurm_info['slurm']['account']
                        cpu_partition = slurm_info['slurm']['cpu_partition']
                    except KeyError as e:
                        print(f"Error: Missing required SLURM configuration: {e}")
                        print("Please ensure your SLURM configuration includes account and cpu_partition.")
                        raise SystemExit(1) from e

                    srun_args = [
                        "srun",
                        f"--account={account}",
                        f"--partition={cpu_partition}",
                        "-t",
                        "90",
                        "--exclusive",
                        "--pty",
                    ]
                    if slurm_info['slurm'].get('cpu_partition_gres'):
                        srun_args.append(f"--gpus-per-node={slurm_info['slurm']['cpu_partition_gres']}")

                    full_cmd_args = srun_args + cmd_args
                    print(f"Submitting download via SLURM: {' '.join(srun_args)} enroot import docker://{image_url}")
                else:
                    full_cmd_args = cmd_args
                    print(f"Downloading {image_url} locally to {source_sqsh_file}")
                    clean_env = None  # Don't use clean env for local execution

                try:
                    subprocess.run(full_cmd_args, check=True, text=True, env=clean_env)
                    if not skip_ca_upgrade:
                        with open(ca_marker, "w") as marker:
                            marker.write("ca-certificates upgraded (CVE-2026-111943)\n")
                    print(f"Successfully downloaded {source_sqsh_file}\n")
                except subprocess.CalledProcessError as e:
                    print(f"Error: Failed to download {source_sqsh_file}: {str(e)}\n")
                    if install_method == 'slurm':
                        print(
                            "Please ensure the compute nodes you selected in the 'CPU_PARTITION' option are configured correctly, or try a local installation (if available)."
                        )
                        # Check if running from a virtual environment
                        if detect_virtual_environment():
                            if has_active_conda_environment():
                                venv_type = 'conda'
                            else:
                                venv_type = 'venv'
                            print(
                                f"Note: Running from a {venv_type} environment. Architecture mismatch between login and compute nodes may cause 'Exec format error'."
                            )
                            print("If using conda, please try venv instead with system python.")
                    else:
                        print("Ensure your system has appropriate resources or try a SLURM-based installation.")
                    raise SystemExit(1) from e

            # If using a shared directory, create/update symlink in the local install
            if shared_images_dir:
                symlink_path = os.path.join(local_images_dir, filename)
                abs_source_sqsh_file = os.path.abspath(source_sqsh_file)

                # Handle existing file/symlink at target
                if os.path.lexists(symlink_path):
                    if os.path.islink(symlink_path) and os.readlink(symlink_path) == abs_source_sqsh_file:
                        continue  # Symlink exists and is correct
                    os.remove(symlink_path)

                os.symlink(abs_source_sqsh_file, symlink_path)
                print(f"✓ Linked {filename} to shared image directory.")

        except Exception as e:
            print(f"Error processing image {image_url}: {str(e)}")
            raise SystemExit(1) from e
