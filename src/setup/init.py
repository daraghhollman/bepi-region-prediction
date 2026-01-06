"""
Initial setup to use this repository.

Downloads hollman_2025_crossing_list.csv

from:
    https://zenodo.org/records/17814795/files/hollman_2025_crossing_list.csv?download=1

to:
    ./resources/hollman_2025_crossing_list.csv

"""

import datetime as dt
import logging
import os
import zipfile
from pathlib import Path

import requests
from dateutil.parser import parse
from tqdm import tqdm

RESOURCES_DIR = Path(__file__).parent / "../../resources/"


def main():

    # Set up logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Some setup for logging colours. See:
    # https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)

    # Ensure directory exists, if not, create it.
    if not os.path.isdir(RESOURCES_DIR):
        os.makedirs(RESOURCES_DIR)

    # Download files
    logger.info("Downloading crossing list...")

    url = "https://zenodo.org/records/17814795/files/hollman_2025_crossing_list.csv?download=1"
    response = requests.get(url)
    with open(f"{RESOURCES_DIR}/hollman_2025_crossing_list.csv", "wb") as file:
        file.write(response.content)

    # Downloading Bepi SPICE repo.
    # The ESA bitbucket site is down as of the writing of this, so for now, we
    # download from https and only update if needed.
    url = "https://spiftp.esac.esa.int/data/SPICE/BEPICOLOMBO/misc/skd/BEPICOLOMBO.zip"
    zip_local_path = RESOURCES_DIR / "bepicolombo-skd.zip"

    if os.path.exists(zip_local_path):
        # If the file exists, we want to check if its been since modified. If
        # so, redownload and extract. Else, skip.
        response = requests.head(url)
        url_time = response.headers["last-modified"]
        url_date = parse(url_time)

        local_date = dt.datetime.fromtimestamp(
            os.path.getmtime(zip_local_path), tz=dt.timezone.utc
        )

        if url_date > local_date:
            logger.info("SPICE kernels out of date. Redownloading...")
            download_bepi_spice(url, zip_local_path)
            extract_bepi_spice(zip_local_path)

    else:
        download_bepi_spice(url, zip_local_path)
        extract_bepi_spice(zip_local_path)

    logger.info("Setup complete")


def extract_bepi_spice(zip_location: Path) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Extracting BepiColombo SPICE kernels")

    with zipfile.ZipFile(zip_location, "r") as zip_file:
        zip_file.extractall(RESOURCES_DIR)

    os.rename(RESOURCES_DIR / "BEPICOLOMBO/", RESOURCES_DIR / "bepicolombo-spice/")


def download_bepi_spice(url: str, download_to: Path) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Downloading most up-to-date SPICE kernels for BepiColombo...")

    download_with_progress_bar(url, download_to)


def download_with_progress_bar(
    url: str, download_to: Path, block_size: int = 1024
) -> None:

    response = requests.get(url, stream=True)

    # Total size of the file in bytes
    total_size = int(response.headers.get("content-length", 0))

    with tqdm(
        total=total_size, unit="B", unit_scale=True, desc=str(download_to)
    ) as progress_bar:

        with open(download_to, "wb") as file:

            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError("Download failed or incomplete!")


class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(levelname)s (%(filename)s:%(lineno)d) - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: format,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


if __name__ == "__main__":
    main()
