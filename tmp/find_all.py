import os
import pickle
file_list = []

for root, dirs, files in os.walk('train'):
    file_list.extend(files)

pickle.dump(files, open('files.pkl', 'wb'))