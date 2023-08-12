import os
import re
import subprocess
import requests


# The deb files we need to install
deb_files_startstwith = [
    "chromium-codecs-ffmpeg-extra_",
    "chromium-codecs-ffmpeg_",
    "chromium-browser_",
    "chromium-chromedriver_",
]


def get_latest_version() -> str:
    # A request to security.ubuntu.com for getting latest version of chromium-browser
    # e.g. "112.0.5615.49-0ubuntu0.18.04.1_amd64.deb"
    url = "http://security.ubuntu.com/ubuntu/pool/universe/c/chromium-browser/"
    r = requests.get(url)
    if r.status_code != 200:
        raise Exception("status_code code not 200!")
    text = r.text

    # Find latest version
    pattern = '<a\shref="chromium\-browser_([^"]+.ubuntu0\.18\.04\.1_amd64\.deb)'
    latest_version_search = re.search(pattern, text)
    if latest_version_search:
        latest_version = latest_version_search.group(1)
    else:
        raise Exception("Can not find latest version!")
    return latest_version


def download_chromium(latest_version: str, quiet: bool):
    deb_files = []
    for deb_file in deb_files_startstwith:
        deb_files.append(deb_file + latest_version)

    for deb_file in deb_files:
        url = f"http://security.ubuntu.com/ubuntu/pool/universe/c/chromium-browser/{deb_file}"

        # Download deb file
        if quiet:
            command = f"wget -q -O /content/{deb_file} {url}"
        else:
            command = f"wget -O /content/{deb_file} {url}"
        print(f"Downloading: {deb_file}")
        os.system(command)

        # Install deb file
        if quiet:
            command = f"apt-get install /content/{deb_file} >> apt.log"
        else:
            command = f"apt-get install /content/{deb_file}"
        print(f"Installing: {deb_file}\n")
        os.system(command)

        # Delete deb file from disk
        os.remove(f"/content/{deb_file}")


def check_chromium_installation():
    try:
        subprocess.call(["chromium-browser"])
        print("Chromium installation successfull.")
    except FileNotFoundError:
        print("Chromium Installation Failed!")


def install_selenium_package(quiet: bool):
    if quiet:
        os.system("pip install selenium - qq >> pip.log")
    else:
        os.system("pip install selenium")
