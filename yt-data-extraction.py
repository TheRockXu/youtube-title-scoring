from selenium.webdriver.firefox.options import Options
from tqdm import tqdm
from selenium import webdriver
import time

def search(query):
    url = 'https://www.youtube.com/results?search_query='+query
    options = Options()
    options.add_argument("--headless")

    driver = webdriver.Firefox(options=options)
    results = []
    try:
        driver.get(url)
        while driver.execute_script('return document.readyState') !='complete': continue
        for i in range(5):
            time.sleep(1)
            if driver.execute_script('return document.readyState') !='complete':
                time.sleep(2)
            driver.execute_script('window.scrollBy(0, 1028)')

        elems = driver.find_elements_by_id('meta')
        for elem in elems:
            try:
                title = elem.find_element_by_id('video-title').get_attribute('title')
                [views, period] = elem.find_element_by_id('metadata-line').text.split('\n')
                results.append({
                    'title':title,
                    'period':period,
                    'views':views
                })
            except Exception as e:
                print(e)
        # items = elem.find_elements_by_tag_name('li')
    except Exception as e:
        driver.close()
        print(e)
    driver.close()
    return results

if __name__ == '__main__':
    import pandas as pd
    terms = open('search_terms', encoding='utf-8').read().split('\n')
    terms = set(terms)
    results = []
    for i in tqdm(terms):
        res = search(i)
        results.extend(res)
    df = pd.DataFrame(results)
    df.to_csv('data.csv')


