"""Download the public raw datasets used by the validation scripts.

The files are fetched from the official UCI and Figshare records documented in
DATASETS.md. Third-party data remain subject to their original licences.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXTERNAL = ROOT / "external_data"

UCI_FILES = {
    "gas_sensor_array_drift.zip": (
        "https://archive.ics.uci.edu/static/public/224/"
        "gas%2Bsensor%2Barray%2Bdrift%2Bdataset.zip"
    ),
    "hydraulic_condition_monitoring.zip": (
        "https://archive.ics.uci.edu/static/public/447/"
        "condition%2Bmonitoring%2Bof%2Bhydraulic%2Bsystems.zip"
    ),
}

AHU_FILES = {
    "office_scientific_data.csv": (
        "https://ndownloader.figshare.com/files/53483432",
        "646799c043c8d0455d507cea223d370f",
    ),
    # Figshare's source file is misspelled as "hosptial"; save the corrected
    # local name expected by run_ahu_field_validation.py.
    "hospital_scientific_data.csv": (
        "https://ndownloader.figshare.com/files/53483435",
        "3be98fc565727053878c205de03ffb4f",
    ),
    "auditorium_scientific_data.csv": (
        "https://ndownloader.figshare.com/files/53483438",
        "044c9faf5d99d7621edd7fb05850dfd0",
    ),
    "FDD_processing.ipynb": (
        "https://ndownloader.figshare.com/files/53483891",
        "c8d6415b7aaca0d93ecaf66937082193",
    ),
}


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_md5: str | None, force: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        if expected_md5 is None or md5(destination) == expected_md5:
            print(f"exists: {destination.relative_to(ROOT)}")
            return

    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "ro-pdrf-data-downloader/1.0"})
    print(f"download: {url}")
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)

    if expected_md5 is not None:
        actual = md5(temporary)
        if actual != expected_md5:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"MD5 mismatch for {destination.name}: expected {expected_md5}, got {actual}"
            )
    temporary.replace(destination)
    print(f"saved: {destination.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="replace existing downloads")
    args = parser.parse_args()

    for name, url in UCI_FILES.items():
        download(url, EXTERNAL / name, None, args.force)
    for name, (url, checksum) in AHU_FILES.items():
        download(url, EXTERNAL / "ahu_field" / name, checksum, args.force)

    print("All public datasets are ready. See DATASETS.md for citation and licence details.")


if __name__ == "__main__":
    main()
