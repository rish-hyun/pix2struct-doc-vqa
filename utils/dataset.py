import os
import sys
import toml
import argparse
from time import sleep

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from .chromium import (
    get_latest_version,
    download_chromium,
    check_chromium_installation,
    install_selenium_package
)


config = toml.load('config.toml')['DATASET']

EMAIL = config.get('EMAIL')
PASSWORD = config.get('PASSWORD')
DATASET_NAME = config.get('DATASET_NAME')
CHROME_DRIVER_PATH = config.get('CHROME_DRIVER_PATH')


def download_from_rrc(save_directory: str = os.getcwd()) -> None:

    if 'google.colab' in sys.modules:
        quiet = True  # verboseness of wget and apt

        latest_version = get_latest_version()
        download_chromium(latest_version, quiet)
        check_chromium_installation()
        install_selenium_package(quiet)

    # Set the Chrome driver options
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    prefs = {"download.default_directory": save_directory}
    options.add_experimental_option("prefs", prefs)

    # Start the Chrome driver
    driver = webdriver.Chrome(
        service=Service(executable_path=CHROME_DRIVER_PATH),
        options=options
    )

    driver.get('https://rrc.cvc.uab.es/?com=contestant')

    driver.find_element(by=By.ID, value='input_email').send_keys(EMAIL)
    driver.find_element(by=By.ID, value='input_password').send_keys(PASSWORD)
    driver.find_element(by=By.CLASS_NAME, value='btn-primary').click()

    locator = (By.CLASS_NAME, "bs-callout")
    WebDriverWait(driver, 3).until(EC.presence_of_element_located(locator))

    driver.get('https://rrc.cvc.uab.es/?ch=17&com=downloads')

    link = driver.find_element(
        by=By.XPATH,
        value=f"//*[contains(text(), '{DATASET_NAME}:')]/a"
    )
    link.click()

    sleep(3)

    # Quit the driver if file downloaded
    while any(filename.endswith('.crdownload') for filename in os.listdir(save_directory)):
        sleep(1)

    driver.quit()


def load_from_gdrive(mount_directory: str = os.getcwd()) -> None:
    from google.colab import drive
    drive.mount(f'/{mount_directory}/drive')


def extract_tarfile(file_name: str, extract_directory: str = os.getcwd()) -> None:
    os.system(f'tar -xf {file_name} -C {extract_directory}')


if __name__ == '__main__':
    ...