import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import warnings
from urllib3.exceptions import InsecureRequestWarning

# Tắt cảnh báo SSL không xác thực
warnings.filterwarnings('ignore', category=InsecureRequestWarning)

# Danh sách User-Agent giả lập trình duyệt để tránh bị chặn
user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
]

headers = {'User-Agent': random.choice(user_agents)}
session = requests.Session()

# Lấy nội dung văn bản từ trang web
def fetch_page_text(url, retries=3, timeout=5):
    attempt = 0
    while attempt < retries:
        try:
            response = session.get(url, timeout=timeout, headers=headers, verify=False)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup.get_text(separator='\n', strip=True)
        except requests.exceptions.RequestException:
            attempt += 1
            time.sleep(1)
    return None

# Crawl dữ liệu từ danh sách URL
def crawl_urls(url_list):
    results = {}
    for index, url in enumerate(url_list):
        print(f"Fetching: {url}")
        text = fetch_page_text(url)
        if text:
            results[url] = text
        else:
            print(f"Failed to crawl: {url}")
    return results

# Lưu tất cả dữ liệu đã crawl vào một file duy nhất
def save_crawled_data(crawled_data, urls, output_file='data/all_data.txt'):
    with open(output_file, 'w', encoding='utf-8') as file:
        for url in urls:
            text = crawled_data.get(url, "")
            cleaned_text = text.replace('\n', ' ')
            file.write(cleaned_text + '\n\n')  # Thêm newline để phân cách các đoạn
    print(f"Saved all data to: {output_file}")


if __name__ == "__main__":
    file_path = './data/data_source.csv'
    data = pd.read_csv(file_path)

    # Nếu chỉ có 1 cột chứa URL thì dùng cột đầu tiên:
    if 'Source URL' in data.columns:
        urls = data['Source URL'].dropna().str.strip().unique()
    else:
        urls = data.iloc[:, 0].dropna().str.strip().unique()

    # Bắt đầu crawl
    crawled_data = crawl_urls(urls)

    # Lưu kết quả
    save_crawled_data(crawled_data, urls)

    print("Crawling complete!")
