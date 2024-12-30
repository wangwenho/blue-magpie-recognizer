import time
import os
import requests
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class EbirdCrawler:
    """
    Scrape and download images from ebird.org
    """

    def __init__(self, url: str, class_name: str, max_image_num: int = 1500):
        self.url = url
        self.class_name = class_name
        self.max_image_num = max_image_num
    
    def scrape_images(self):
        """
        Scrape images from ebird.org
        """

        driver = webdriver.Edge()

        driver.get(self.url)

        load_more_button_selector = '#content > div > div > form > div.pagination > button'

        count = 0
        # click the load more button until it disappears
        try:
            while count <= self.max_image_num / 30:
                load_more_button = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, load_more_button_selector))
                )
                ActionChains(driver).move_to_element(load_more_button).click(load_more_button).perform()
                time.sleep(0.1)
                count += 1
        except:
            print('loaded all images. start scraping...')

        # get all images
        image_elements = driver.find_elements(By.TAG_NAME, 'img')
        self.download_img(image_elements)
        
        driver.quit()

    def download_img(self, image_elements):
        """
        download images with given image elements
        """

        os.makedirs(f"raw_images/{self.class_name}", exist_ok=True)
        
        # download images
        print(f"start downloading {len(image_elements)} images...")
        for i, img in enumerate(tqdm(image_elements)):
            img_url = img.get_attribute('src')
            if img_url:
                response = requests.get(img_url)
                with open(f"raw_images/{self.class_name}/{str(i + 1).zfill(4)}.jpg", "wb") as file:
                    file.write(response.content)