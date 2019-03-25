# to unzip the .tgz files to same folder



import os
import tarfile

raw_folder = '/home/user/vitalv/voice-gender-classifier/raw'

for f in os.listdir(raw_folder):
    if f.endswith('.tgz'):
        tar = tarfile.open(os.path.join(raw_folder, f))
        tar.extractall(raw_folder)
tar.close()
