from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from pathlib import Path
import pandas as pd
import numpy as np

data = []
csv = []

def build_questions_and_answers(cols):
    q1 = "What is " + cols[0].text +"?"
    if (cols[2].text == "Classification"):
        a1 = cols[0].text + " is a model of machine learning used for classification."
        a4 = cols[0].text + " is a model of machine learning used for regression."
        data.append([q1, a4, -0.9 + np.random.normal(0,0.1)])
    elif (cols[2].text == "Regression"):
        a1 = cols[0].text + " is a model of machine learning used for regression."
        a4 = cols[0].text + " is a model of machine learning used for classification."
        data.append([q1, a4, -0.9 + np.random.normal(0,0.1)])
    else:
        a1 = cols[0].text + " is a model of machine learning used for classification and regression."
    q2 = "How i use " + cols[0].text + "?"
    a2 = cols[0].text + " is implemented in Caret under the library " + cols[3].text + ". The method in Caret" + " to use this model is " + cols[1].text + "."
    q3 = "What are the tuning parameters of " + cols[0].text + "?"
    a3 = cols[0].text + " have the following tuning parameters: " + cols[4].text + "."
    data.append([q1, a1, 0.9 + np.random.normal(0,0.1)])
    data.append([q2, a2, 0.9 + np.random.normal(0,0.1)])
    data.append([q3, a3, 0.9 + np.random.normal(0,0.1)])
    csv.append([cols[0].text, a1+" "+a2+" "+a3])
    

driver = webdriver.Firefox()

site = 'https://topepo.github.io/caret/available-models.html'
driver.get(site)


#The table is loaded dynamically with an AJAX jquery, so we wait until the table is loaded.
WebDriverWait(driver,10).until(EC.visibility_of_element_located((By.ID,"DataTables_Table_0")))
table_id = driver.find_element(By.ID, "DataTables_Table_0")
tbody = table_id.find_element(By.TAG_NAME,'tbody')


rows = tbody.find_elements(By.TAG_NAME,"tr")

for row in rows:
    cols = row.find_elements(By.TAG_NAME, "td")
    build_questions_and_answers(cols)

driver = webdriver.Firefox()

with open("/Users/melland/Universidad/CollaborationProject/cran_names.txt", 'r') as f:
    lines = f.readlines()

for line in lines:
    url = 'https://cran.r-project.org/web/packages/'+line.strip() +'/index.html'
    driver.get(url)
    ps = driver.find_elements_by_xpath("/html/body/div/p")
    data.append(['What is the ' + line.strip() + ' package?', ps[0].text, 0.9 + np.random.normal(0,0.1)])

df1 = pd.DataFrame(csv, columns=['Title','Description'])
df2 = pd.DataFrame(data, columns =['Question', 'Answer', 'Score'])

df2.to_csv(str(Path(__file__).parent) + '/caretQuestions.csv', index=False, header=True)


