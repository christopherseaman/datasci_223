from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from pathlib import Path
import os
import tempfile


def setup_driver():
    chrome_options = Options()
    # Create a temporary directory for Chrome profile
    temp_dir = tempfile.mkdtemp()
    chrome_options.add_argument(f"--user-data-dir={temp_dir}")
    chrome_options.add_argument("--window-size=1920,1080")
    return webdriver.Chrome(options=chrome_options)


def take_screenshot(driver, url, filename, wait_for=None):
    driver.get(url)
    if wait_for:
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, wait_for))
            )
        except:
            # If element not found, wait a bit and take screenshot anyway
            time.sleep(5)
    time.sleep(2)  # Allow for any animations
    driver.save_screenshot(f"media/{filename}")


def main():
    # Create media directory
    Path("media").mkdir(exist_ok=True)

    driver = setup_driver()

    try:
        # Create New Repository
        take_screenshot(
            driver,
            "https://github.com/new",
            "github_create_repo.png",
            wait_for=".repository-content",
        )

        # Example Repository (using your repository)
        take_screenshot(
            driver,
            "https://github.com/christopherseaman/datasci_223",
            "github_example_repo.png",
            wait_for=".repository-content",
        )

        # GitHub Pages Settings
        take_screenshot(
            driver,
            "https://github.com/christopherseaman/datasci_223/settings/pages",
            "github_pages_settings.png",
            wait_for="#pages-source",
        )

        # GitHub Actions
        take_screenshot(
            driver,
            "https://github.com/christopherseaman/datasci_223/actions",
            "github_actions.png",
            wait_for=".workflow-run",
        )

        # Published Site
        take_screenshot(
            driver,
            "https://christopherseaman.github.io/datasci_223/",
            "github_published_site.png",
            wait_for="body",
        )

    finally:
        driver.quit()


if __name__ == "__main__":
    main()
