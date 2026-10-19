# ==================== SAMPLE 1 ====================
# Now convert the obtained dictionary to a CSV file via pandas module
df = pd.DataFrame(scraped_data)
df.to_csv('internships.csv', index=False)


import shlex
import requests
from subprocess import Popen, PIPE


def execute_and_return(cmd):
    """
    Execute the external command and get its exitcode, stdout and stderr.
    """

# ==================== SAMPLE 2 ====================
args = parser.parse_args()

in1 = ffmpeg_args()

in1 = ffmpeg.input(args.inputfile)

v1 = in1.filter('trim', start=float(args.starttime), end=(args.endtime))
v2 = in1.filter('trim', start=float(args.endtime))

out1 = ffmpeg.output(v1, args.outputfile1)
out2 = ffmpeg.output(v2, args.outputfile2)

out1.run()


from selenium import webdriver
import requests as rq
import os
from bs4 import BeautifulSoup
import time

# path= E:\web scraping\chromedriver_win32\chromedriver.exe
path = input("Enter Path : ")

url = input("Enter URL : ")

output = "output"


def get_url(path, url):
    driver = webdriver.Chrome(executable_path=r"{}".format(path))
    driver.get(url)
    print("loading.....")
    res = driver.execute_script("return document.documentElement.outerHTML")

    return res


def get_img_links(res):
    soup = BeautifulSoup(res, "lxml")
    imglinks = soup.find_all("img", src=True)
    return imglinks


def download_img(img_link, index):
    try:
        extensions = [".jpeg", ".jpg", ".png", ".gif"]
        extension = ".jpg"
        for exe in extensions:
            if img_link.find(exe) > 0:
                extension = exe
                break

        img_data = rq.get(img_link).content
        with open(output + "\\" + str(index + 1) + extension, "wb+") as f:
            f.write(img_data)
        
        f.close()
    except Exception:
        pass


result = get_url(path, url)
time.sleep(60)
img_links = get_img_links(result)
if not os.path.isdir(output):
    os.mkdir(output)

for index, img_link in enumerate(img_links):
    img_link = img_link["src"]
    print("Downloading...")
    if img_link:
        download_img(img_link, index)
print("Download Complete!!")


import glob
import pandas as pd

extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
combined_csv.to_csv( "combined_csv.csv.csv", index=False, encoding='utf-8-sig')


import requests
import json
import time
    
    
# call API
def get_baidu_poi(roi_key, city_str, baidu_ak, output):
    """
    inputs:
        roi_key: poi name
        city_str: city name
        baidu_ak: baidu web API AK
        output: file save path
    """
    now_time = time.strftime("%Y-%m-%d")
    page_num = 0
    logfile = open(output + "/" + now_time + ".log", "a+", encoding="utf-8")
    file = open(output + "/" + now_time + ".txt", "a+", encoding="utf-8")

# ==================== SAMPLE 3 ====================
print(img.shape)

# set the ratio of resized image
k = 5
width = int((img.shape[1])/k)
height = int((img.shape[0])/k)

# resize the image by resize() function of openCV library
scaled = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
print(scaled.shape)

# show the resized image using imshow() function of openCV library
cv2.imshow("Output", scaled)
cv2.waitKey(500)
cv2.destroyAllWindows()

# get the resized image output by imwrite() function of openCV library
cv2.imwrite('resized_output_image.jpg', scaled)


# importing socket library
import socket

def get_hostname_IP():
    hostname = input("Please enter website address(URL):")
    try:
        print (f'Hostname: {hostname}')
        print (f'IP: {socket.gethostbyname(hostname)}')
    except socket.gaierror as error:
        print (f'Invalid Hostname, error raised is {error}')

get_hostname_IP()



""" 
THis program is capable of converting from one currency to another as of today itself. 
It uses the api at fixer.io and then calculates the value of the currency in terms of the other as of today.
"""

# https://github.com/chavarera/python-mini-projects/issues
# https://medium.com/@cereblanco/setup-black-and-isort-in-vscode-514804590bf9
# Source: https://fixer.io/quickstart
# Imp read: https://stackoverflow.com/questions/3139879/how-do-i-get-currency-exchange-rates-via-an-api-such-as-google-finance


import requests 
import json
import sys
from pprint import pprint

# The below 4 lines bring out the value of currency from the api at fixer.io.  I had to register there, the key is unique to me.
url = "http://data.fixer.io/api/latest?access_key=33ec7c73f8a4eb6b9b5b5f95118b2275"
data = requests.get(url).text
data2 = json.loads(data) #brings whether request was successful,timestamp etc
fx = data2["rates"]

# ==================== SAMPLE 4 ====================
with open(filename, 'r') as f:
    for line in f:
        # if case is ignored then Great and great are same words
        list_of_words.extend(re.findall(r"[\w]+", line.lower()))
        # else use this alternate method:
        # list_of_words.extend(re.findall(r"[\w]+", line))

        
# Creating a dictionary to store the number of occurence of a word
unique = {}
for each in list_of_words:
    if each not in unique:
        unique[each] = 0
    unique[each] += 1
 
# Creating a list to sort the final unique words
s = []

# If occurence of a word(val) is 1 then it is unique
for key, val in unique.items():
    if val == 1:
        s.append(key)
        
print(sorted(s))
    



# Import Required Library
from tkinter import *
import datetime
import time
import winsound
from threading import *

# Create Object
root = Tk()

# Set geometry
root.geometry("400x200")

# Use Threading
def Threading():
	t1=Thread(target=alarm)
	t1.start()

def alarm():
	# Infinite Loop
	while True:
		# Set Alarm
		set_alarm_time = f"{hour.get()}:{minute.get()}:{second.get()}"

		# Wait for one seconds
		time.sleep(1)

		# Get current time
		current_time = datetime.datetime.now().strftime("%H:%M:%S")
		print(current_time,set_alarm_time)

		# Check whether set alarm is equal to current time or not
		if current_time == set_alarm_time:
			print("Time to Wake up")
			# Playing sound
			winsound.PlaySound("sound.wav",winsound.SND_ASYNC)

# Add Labels, Frame, Button, Optionmenus
Label(root,text="Alarm Clock",font=("Helvetica 20 bold"),fg="red").pack(pady=10)
Label(root,text="Set Time",font=("Helvetica 15 bold")).pack()

frame = Frame(root)
frame.pack()

hour = StringVar(root)
hours = ('00', '01', '02', '03', '04', '05', '06', '07',
		'08', '09', '10', '11', '12', '13', '14', '15',
		'16', '17', '18', '19', '20', '21', '22', '23', '24'
		)
hour.set(hours[0])

hrs = OptionMenu(frame, hour, *hours)
hrs.pack(side=LEFT)

minute = StringVar(root)

# ==================== SAMPLE 5 ====================
def display_header(header):
    print("")
    print("###########################".center(os.get_terminal_size().columns))
    print(f"###### {header} ######".center(os.get_terminal_size().columns))
    print("###########################".center(os.get_terminal_size().columns))

if __name__ == "__main__":
    display_header("Alarm Program")
    set_alarm()

from bs4 import BeautifulSoup
import requests

# Trying to open a random wikipedia article
# Special:Random opens random articles
res = requests.get("https://en.wikipedia.org/wiki/Special:Random")
res.raise_for_status()

# pip install htmlparser
wiki = BeautifulSoup(res.text, "html.parser")

r = open("random_wiki.txt", "w+", encoding='utf-8')

# Adding the heading to the text file
heading = wiki.find("h1").text

r.write(heading + "\n")
for i in wiki.select("p"):
    # Optional Printing of text
    # print(i.getText())
    r.write(i.getText())

r.close()
print("File Saved as random_wiki.txt")



# !/usr/bin/env python
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import json
import requests


# article url
# URL = "https://www.geeksforgeeks.org/what-can-i-do-with-python/"


def get_driver():
    # chrome options settings
    chrome_options = webdriver.ChromeOptions()
    settings = {
        "recentDestinations": [
            {"id": "Save as PDF", "origin": "local", "account": ""}
        ],
        "selectedDestinationId": "Save as PDF",
        "version": 2,
    }
    prefs = {
        "printing.print_preview_sticky_settings.appState": json.dumps(settings)
    }
    chrome_options.add_experimental_option("prefs", prefs)
    chrome_options.add_argument("--kiosk-printing")

    # launch browser with predefined settings
    browser = webdriver.Chrome(
        executable_path=ChromeDriverManager().install(), options=chrome_options
    )
    return browser


def download_article(URL):
    browser = get_driver()
    browser.get(URL)

    # launch print and save as pdf
    browser.execute_script("window.print();")
    browser.close()


if __name__ == "__main__":
    URL = input("provide article URL: ")
    # check if the url is valid/reachable
    if requests.get(URL).status_code == 200:
        try:
            download_article(URL)
            print("Your article is successfully downloaded")
        except Exception as e:
            print(e)
    else:
        print("Enter a valid working URL")
