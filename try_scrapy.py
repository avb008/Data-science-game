# -*- coding: utf-8 -*-
"""
Created on Wed May 24 21:36:03 2017

@author: vushesh
"""

import scrapy
import json
import time
#from urllib.request import urlopen #run this if u have python3
#from urllib2 import urlopen #run this if u have python2.7

with open('D:/Studies/Online Competitions/DSG/songs/songs.json','r') as s:
	songs=json.loads(s.read())

songs = songs[100000:200000]
class QuotesSpider(scrapy.Spider):
    name = "quotes"
    
    def start_requests(self):
        urls = []
        for idx, song_id in enumerate(songs): 
            x = 'https://api.deezer.com/track/{}/'.format(song_id)
            urls.append(x)
        count = 0
        for x in urls:
            yield scrapy.Request(url=x, callback=self.parse )
            count += 1
            if count%50 == 0:
                time.sleep(3.1)
            
    
    def parse(self, response):
        page = response.url.split("/")[-2]
        filename = 'song%s.json' % page
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log('Saved file %s' % filename)

"""        
def start_requests(self): 
while n<len(songs)-50:
            urls = ['https://api.deezer.com/track/{}/'.format(id_) for id_ in songs[n:n+50]] 
            for x in urls:
                yield scrapy.Request(url=x, callback=self.parse)
            time.sleep(5)
            n+=50
            
            """