import requests
from bs4 import BeautifulSoup

Link=requests.get("https://en.wikipedia.org/wiki/Deep_learning").text
b=BeautifulSoup(Link,"html.parser")
Title=b.find('title').string;
print(Title)

outfile=open('output.txt','a+',encoding='utf-8')
atags=b.find_all('a');
for atag in  atags:
    if atag.get('href'):
        outfile.write(str(atag.get('href'))+'\n')
        print(atag.get('href'))