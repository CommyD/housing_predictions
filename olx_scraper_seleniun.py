import csv
import json
import os.path
import random

from selenium import webdriver
from selenium.webdriver import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
import pandas as pd
import time
from pathlib import Path
from tqdm import tqdm
import constants


def start_driver():
    options = Options()
    # options.add_argument('--headless')  # fără UI
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=options)
    return driver


def parse_listings_from_storia(driver):
    time.sleep(4)
    try:
        accept_button = driver.find_element(By.ID,"onetrust-accept-btn-handler")
        accept_button.click()
    except NoSuchElementException:
        print("no accept buttom")
    listings = driver.find_elements(By.CSS_SELECTOR, ".css-xv0nyo")
    data = []

    for l in listings:
        try:
            title = l.find_element(By.CLASS_NAME, "css-135367").get_attribute("innerText")
            link = l.find_element(By.TAG_NAME, "a").get_attribute("href")
            price = l.find_element(By.CLASS_NAME, "css-1grq1gi").get_attribute("innerText")
            location = l.find_element(By.CLASS_NAME, "css-42r2ms").get_attribute("innerText")
            # size = l.find_element(By.CSS_SELECTOR, ".css-17je0kd").get_attribute("innerText")
            data.append({
                constants.TITLE: title,
                constants.LINK: link,
                constants.PRICE: price,
                constants.LOCATION: location,
                # constants.SIZE: size,
            })
        except NoSuchElementException as e:
            print(f"Error getting details for {l} error={e}")
            continue

    return data

def extract_details_from_ad(driver, link):
    # hostname = driver.current_url  # modificare 18.07, urmatoarele 3 linii adaugate
    driver.get(link)
    time.sleep(random.uniform(3, 6))
    hostname = driver.current_url

    details = {
        constants.SIZE: None,
        constants.ROOMS: None,
        constants.FLOOR: None,
        constants.YEAR: None,
        constants.HEATING: None,
        constants.ELEVATOR: None,
        constants.SELLER: None,
        constants.APPARTMENT_TYPE: None,
        constants.BUILDING_TYPE: None,
        constants.WINDOWS_TYPE: None,
        constants.SOURCE: "storia" if "storia.ro" in hostname else "olx"
    }

    if "storia.ro" in hostname:
        # return parse_storia_details(driver)  #modificare 18.07
        details.update(parse_storia_details(driver))
        return details


    return details

def parse_storia_details(driver):
    result = {
        constants.SIZE: None,
        constants.ROOMS: None,
        constants.FLOOR: None,
        constants.YEAR: None,
        constants.HEATING: None,
        constants.ELEVATOR: None,
        constants.SELLER: None,
        constants.APPARTMENT_TYPE: None,
        constants.BUILDING_TYPE: None,
        constants.WINDOWS_TYPE: None,
        constants.SOURCE: "storia"
    }
    time.sleep(4)
    try:
        accept_button = driver.find_element(By.ID,"onetrust-accept-btn-handler")
        accept_button.click()
    except NoSuchElementException:
        print("no accept buttom")
    # driver.find_element(By.CSS_SELECTOR,"button.css-8cx88i:nth-child(1)").send_keys(Keys.SPACE)
    time.sleep(3)
    driver.find_element(By.CSS_SELECTOR, "button.css-8cx88i").send_keys(Keys.SPACE)


    try:
        spec_boxes = driver.find_elements(By.CLASS_NAME, "css-1xw0jqp")  # css-8mnxk5 e4rbt3a0
        print(f"found {len(spec_boxes)} boxes")
        for box in spec_boxes:
            paragraphs = box.find_elements(By.TAG_NAME, "p")
            texts = [p.get_attribute("innerText") for p in paragraphs]
            label, value = texts
            print(texts, len(texts))
            match label:
                case "Suprafață utilă:":
                    result[constants.SIZE] = value
                case "Numărul de camere:":
                    result[constants.ROOMS] = value
                case "Etaj:":
                    result[constants.FLOOR] = value
                case "Anul construcției:":
                    result[constants.YEAR] = value
                case "Încălzire:":
                    result[constants.HEATING] = value
                case "Lift:":
                    result[constants.ELEVATOR] = value
                case "Tip proprietate:":
                    result[constants.APPARTMENT_TYPE] = value
                case "Tip vânzător:":
                    result[constants.SELLER] = value
                case "Tip clădire:":
                    result[constants.BUILDING_TYPE] = value
                case "Tip geamuri:":
                    result[constants.WINDOWS_TYPE] = value
                case _:
                    pass
    except Exception:
        print(f"Error parsing storia")

    return result

def save_state(state_dict):
    print(f'Saving state {state_dict}')
    with open('state.json', 'wt') as file:
        json.dump(state_dict, file)

def load_state():
    with open('state.json', 'rt') as file:
        return json.load(file)

def scrape_olx(n_pages=640):
    driver = start_driver()
    time.sleep(5)
    all_results = []

    for page in range(1, n_pages + 1):
        # url = f"https://www.olx.ro/imobiliare/apartamente-garsoniere-de-vanzare/bucuresti-ilfov-judet/?currency=EUR&page={page}"
        url = f"https://www.storia.ro/ro/rezultate/vanzare/apartament/bucuresti?page={page}"
        print(f"[INFO] Accessing {url}")
        driver.get(url)
        time.sleep(random.uniform(3,6))
        # page_data = parse_listings_from_olx(driver)
        page_data = parse_listings_from_storia(driver)

        print(f"[INFO] Found {len(page_data)} ads on page {page}")
        all_results.extend(page_data)

    df = pd.DataFrame(all_results)

    if not df.empty:
        print("[INFO] Extracting ad details...")

        last_idx = -1
        if os.path.exists('state.json'):
            state = load_state()
            last_link = state.get("last_link")
            matches = df.index[df[constants.LINK] == last_link].tolist()
            if matches:
                last_idx = matches[0]

        write_header = not os.path.exists("raw_listings.csv")

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            if idx <= last_idx:
                continue

            link = row[constants.LINK]
            try:
                details = extract_details_from_ad(driver, link)
                for key, value in details.items():
                    df.at[idx, key] = value

                with open("raw_listings.csv", "a", newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow([
                            constants.TITLE,
                            constants.LINK,
                            constants.PRICE,
                            constants.LOCATION,
                            constants.SIZE,
                            constants.ROOMS,
                            constants.FLOOR,
                            constants.YEAR,
                            constants.HEATING,
                            constants.ELEVATOR,
                            constants.SELLER,
                            constants.APPARTMENT_TYPE,
                            constants.BUILDING_TYPE,
                            constants.WINDOWS_TYPE,
                            constants.SOURCE
                        ])
                        write_header = False  # Doar o dată

                    writer.writerow([
                        row.get(constants.TITLE),
                        row.get(constants.LINK),
                        row.get(constants.PRICE),
                        row.get(constants.LOCATION),
                        details.get(constants.SIZE),
                        details.get(constants.ROOMS),
                        details.get(constants.FLOOR),
                        details.get(constants.YEAR),
                        details.get(constants.HEATING),
                        details.get(constants.ELEVATOR),
                        details.get(constants.SELLER),
                        details.get(constants.APPARTMENT_TYPE),
                        details.get(constants.BUILDING_TYPE),
                        details.get(constants.WINDOWS_TYPE),
                        details.get(constants.SOURCE),
                    ])

                save_state({'last_link': link})
            except Exception as e:
                print(f"Error extracting ad at {link}: {e}")
                continue

    driver.quit()
    return df

if __name__ == "__main__":
    df = scrape_olx()
    df.to_csv("/Users/cosmindanaita/PycharmProjects/AI3/CosminD/housing_predictions//raw_listings.csv", index=False)
    print(f"[DONE] Saved {len(df)} ads in /raw_listings.csv")
