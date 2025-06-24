import csv
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup

# === НАСТРОЙКИ ===
START_URLS = [
    'https://thermex.ru/search/',
]
LOAD_MORE_CLASSES = [
    'ButtonLoadMore_show-more-button__FXKXn',
    'load-more',  # Добавь сюда классы с других сайтов
]
INPUT_FILE = "art.txt"
OUTPUT_CSV = 'dataset_output.csv'
MAX_QUERIES = 1000

# === CSV: СОЗДАНИЕ И ЗАГОЛОВКОВ ===
def init_csv():
    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Site', 'Name', 'Kitchen', 'Time', 'Rate', 'Price', 'Metro'])

# === ОБРАБОТЧИК "ЗАГРУЗИТЬ ЕЩЁ" ===
def load_all_content_by_classes(driver, class_list, max_clicks=50, wait_time=2):
    for class_name in class_list:
        print(f"[INFO] Проверка кнопки с классом: {class_name}")
        for i in range(max_clicks):
            try:
                button = driver.find_element(By.CLASS_NAME, class_name)
                if button.is_displayed() and button.is_enabled():
                    driver.execute_script("arguments[0].scrollIntoView();", button)
                    button.click()
                    print(f"[INFO] 🔁 Клик по кнопке загрузки ({i + 1}) — {class_name}")
                    time.sleep(wait_time)
                else:
                    print("[INFO] Кнопка найдена, но неактивна.")
                    break
            except NoSuchElementException:
                print(f"[INFO] Кнопка '{class_name}' не найдена. Переход к следующей.")
                break
            except ElementClickInterceptedException:
                print("[WARN] Не удалось кликнуть. Пауза.")
                time.sleep(wait_time)
        time.sleep(1)  # Пауза между классами

# === ПРОЧЕЕ ===
def close_popups(driver):
    try:
        popup = driver.find_element(By.CLASS_NAME, 'widget--close')
        popup.click()
        print("[INFO] Popup закрыт.")
    except NoSuchElementException:
        pass
    except ElementClickInterceptedException:
        print("[WARN] Не удалось кликнуть по popup.")

def perform_search(driver, query):
    try:
        search_box = driver.find_element(By.CLASS_NAME, "v-search-global-input")
        search_box.clear()
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)
        print(f"[INFO] Поиск: {query}")
    except Exception as e:
        raise RuntimeError(f"Ошибка ввода запроса: {e}")

def scroll_down(driver, delay=2):
    last_height = driver.execute_script("return document.body.scrollHeight")
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(delay)
    new_height = driver.execute_script("return document.body.scrollHeight")
    return new_height != last_height

def extract_data(html_code, site_label):
    soup = BeautifulSoup(html_code, 'html.parser')
    items = soup.find_all('div', class_='CardTwoBlock_card__GFR2L Template_card__ZnZUD Listing_card__hNL_B')

    data = []
    for block in items:
        soup_block = BeautifulSoup(str(block), 'html.parser')

        name = soup_block.find('span', class_='Text_text__e9ILn Template_title__3vIhm')
        kitchen = soup_block.find('span', class_='Text_text__e9ILn TypeAndPrice_type-url__5phPv')
        time_block = soup_block.find('p', class_='Text_text__e9ILn ScheduleAndPrice_schedule__Rv03e')
        rate = soup_block.find('span', class_='Rating_rating__NLDVH')
        metro = soup_block.find('div', class_='Place_metro__sSZ56')

        active_prices = soup_block.find_all('span', {'data-active': 'true'})
        count = len(active_prices)
        if count == 1:
            price = '700'
        elif count == 2:
            price = '700 - 1700'
        elif count == 3:
            price = '1700 - 3000'
        elif count == 4:
            price = '3000'
        else:
            price = ''

        metro_text = "; ".join([item.get_text(strip=True) for item in metro.find_all('span')]) if metro else None

        row = [
            site_label,
            name.get_text(strip=True) if name else None,
            kitchen.get_text(strip=True) if kitchen else None,
            time_block.get_text(strip=True) if time_block else None,
            rate.get_text(strip=True) if rate else None,
            price,
            metro_text
        ]
        data.append(row)
    return data

def log_to_file(filename, message):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

# === ОБРАБОТКА ОДНОГО САЙТА ===
def process_site(site_url, queries, max_queries):
    print(f"\n=== ПАРСИМ САЙТ: {site_url} ===\n")
    options = Options()
    # options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    driver.maximize_window()
    driver.get(site_url)

    for idx, query in enumerate(queries):
        if idx >= max_queries:
            break
        try:
            close_popups(driver)
            perform_search(driver, query)
            time.sleep(3)

            load_all_content_by_classes(driver, LOAD_MORE_CLASSES)

            html_code = driver.page_source
            results = extract_data(html_code, site_url)

            with open(OUTPUT_CSV, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerows(results)

            log_to_file("log_success.txt", f"[{site_url}] ✅ {idx+1}/{max_queries} — {query} — {len(results)} записей")
            print(f"[SUCCESS] {query} → {len(results)} записей")
        except Exception as e:
            log_to_file("log_errors.txt", f"[{site_url}] ❌ {idx+1}/{max_queries} — {query} — {e}")
            print(f"[ERROR] {query}: {e}")
            continue

    driver.quit()

# === ЗАПУСК ===
if __name__ == "__main__":
    init_csv()
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f.readlines()]

    for site in START_URLS:
        process_site(site, queries, MAX_QUERIES)

    print("\n[INFO] ✅ Все сайты обработаны.")
