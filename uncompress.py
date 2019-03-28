import requests
from bs4 import BeautifulSoup
import re
import shutil
import os

to download dataset from url

URL = "http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/"


def download_file(from_url, local_path):
    r = requests.get(from_url, stream=True)
    with open(local_path, 'wb') as f:
        shutil.copyfileobj(r.raw, f)

def batch_download(matches):
    for match in matches:
        file_url = os.path.join(URL, match['href'])
        file_local = os.path.join('raw', match['href'])
        download_file(file_url, file_local)

def main():

    response = requests.get(URL)
    soup = BeautifulSoup(response.text, 'html.parser')
    matches = soup.find_all('a', attrs={"href": re.compile("tgz")})

    if not os.path.exists('raw'): os.mkdir('raw')
    #raw_folder = os.path.join(__file__, 'raw')
    #raw_folder = os.path.join('.', 'raw')

    batch_download(matches)
















# to unzip the .tgz files to same folder



import os
import tarfile

raw_folder = '/home/user/vitalv/voice-gender-classifier/raw'

for f in os.listdir(raw_folder):
    if f.endswith('.tgz'):
        tar = tarfile.open(os.path.join(raw_folder, f))
        tar.extractall(raw_folder)
tar.close()
