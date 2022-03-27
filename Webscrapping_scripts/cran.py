import re
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from pathlib import Path
import pandas as pd

ml = []
regex = r'<li><a href=\"\.\..*\">(.*)</a></li>'
with open('/Users/melland/Universidad/CollaborationProject/caret/cran_links.txt','r') as f:
    lines = f.readlines()
    

for line in lines:
    print(line)
    ml.append(re.search(regex, line).group(1))

with open('cran_names.txt', 'w') as f:
    f.write("\n".join(ml))
