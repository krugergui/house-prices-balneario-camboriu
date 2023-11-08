# Imports

import traceback
import numpy as np
import time
import db_functions
import re
import logging

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException

from keys import private

# Only display possible problems
selenium_logger = logging.getLogger('selenium.webdriver.remote.remote_connection')
selenium_logger.setLevel(logging.WARNING)

# Webdriver class creation

class WebDriver:
	iconic_places = {}
	db, db_cursor = '', ''

	def __init__(self, headless: bool=True) -> None:
		self.PATH = 'C:\\Users\\Kriggs\\Documents\\Python\\chromedriver.exe'
		self.options = Options()

		if headless:
			self.options.add_argument("--headless")
		
		# Loads uBlock extension
		self.options.add_argument(f'--load-extension=C:\\Users\\{private.USER}\\AppData\\Local\\Google\\Chrome\\User Data\\Default\\Extensions\\cjpalhdlnbpafiamejdnhcphjbkeiagm\\1.50.0_0')
		self.options.add_argument("--start-maximized")
		self.options.add_argument("--log-level=5")
		self.options.add_argument("--disable-smooth-scrolling")

		self.db, self.db_cursor = db_functions.connect_to_database_mysql_connector()

		self.driver = webdriver.Chrome(self.PATH, options=self.options)

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_value, tb):
		self.db.close()
		self.driver.quit()

	def random_sleep(self, short: bool=False, long: bool=False) -> None:
		multiplicator = 1
		if short:
			multiplicator = 0.2
		if long:
			multiplicator = 5
		time.sleep(0.2 + multiplicator * np.random.rand())

	# Hacky, but no official API for this
	def is_browser_alive(self):
		try:
			self.driver.current_url
			return True
		except:
			return False
		
	def open_webpage(self, webpage: str) -> None:
		self.driver.get(webpage)
		self.driver.maximize_window()
		self.random_sleep()

	def scroll_to_bottom(self) -> None:
		self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
		self.random_sleep(long=True)
		self.driver.execute_script("window.scrollTo(0, 0)")