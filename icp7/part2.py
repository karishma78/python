import requests
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/Google"
resp = requests.get(url)
soup = BeautifulSoup(resp.text, 'html.parser')
data = soup.find('div', {'class': 'mw-parser-output'})
with open('input.txt', 'w', encoding='utf-8') as f:
    f.write(str(data.text))