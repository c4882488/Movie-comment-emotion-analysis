from ckiptagger import WS, POS, NER
from selenium import webdriver
from bs4 import BeautifulSoup
import csv

URL = "https://movies.yahoo.com.tw/movieinfo_review.html/id=11215?sort=update_ts&order=asc"
driver = webdriver.Edge(executable_path='./post/edge/msedgedriver')

driver.get(URL)
pageSource = driver.page_source
soup = BeautifulSoup(pageSource, "lxml")

pages = soup.select('.page_numbox li')
info = soup.select('.inform_r')[0]
pages=[i.find('a').text for i in pages if i.find('a') != None][:-1]
pages = [int(i) for i in pages]

MociesName = info.find("h1",class_="inform_title").text
MociesName = [i.strip() for i in MociesName.split("\n") if i !=""]

array = ""
fp = open("post_data/movies_"+MociesName[0]+".csv", "w", newline='',encoding='utf-8')
pos = open("post_data/pos/movies_9.csv", "w", newline='',encoding='utf-8')
eng = open("post_data/eng/movies_9.csv", "w", newline='',encoding='utf-8')
writer = csv.writer(fp, delimiter=',')
writerpos = csv.writer(pos, delimiter=',')
writereng = csv.writer(eng, delimiter=',')

ws = WS("./data")
for i in range(1,max(pages)+1):
    driver.get(URL+"&page="+str(i))
    pageSource = driver.page_source
    soup = BeautifulSoup(pageSource, "lxml")
    items = soup.select('.usercom_list li')

    for i,d in enumerate(items):
        text = d.select("form span")[2].text
        datetime = d.select(".user_time")[0].text[5:]
        date = datetime[:10]
        times = datetime[11:]
        name = d.select(".user_id")[0].text[4:]
        good = d.select(".good span")[0].text[:-2]
        bad = d.select(".bad span")[0].text[1:-2]
        star = d.find_all("div",class_="starovr", attrs={'style':'width: 15px;'})
        ws_results = ws([text])
        t = ""
        for j in ws_results[0]:
            t += j+" "
        if len(star) > 4:
            writerpos.writerow([date,times,t,1])
        else:
            writereng.writerow([date,times,t,0])
        writer.writerow([date,times,t,name,good,bad,str(len(star))])
        print(t)

fp.close()
pos.close()
eng.close()