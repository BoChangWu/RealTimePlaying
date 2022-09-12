import os 
import time
from mido import MidiFile
from math import sqrt
import numpy as np
import tensorflow as tf


seq_len = 256
gen_len = int(sqrt(seq_len))

############## preprocessing ################

paths = []
songs = []

# walk through every file in target folder including itself
# edit your own data path
for r,d,f in os.walk('../data/magenta'):
    '''
    r: route
    d: folder in this folder
    f: file in this folder
    '''

    # if the file is mid file then append its path to paths[]
    for file in f:
        if '.mid' in file:
            paths.append(os.path.join(r,file))

# for each path in the array, create a Mido object and append it to songs
for path in paths:
    mid = MidiFile(path,type=1)
    songs.append(mid)


notes = []
dataset = []
chunk = []

# for each in midi object in list of songs
for i in range(len(songs)):
    for msg in songs[i]:
        # filtering out meta messages
        if not msg.is_meta:
            # filtering out control changes
            if (msg.type =='note_on'):
                notes.append(msg.note)

    for j in range(1,len(notes)):
        chunk.append(notes[j])
        # save each 16 note chunk
        if (j% seq_len==0):
            dataset.append(chunk)
            chunk = []
    print(f"Processing {i} Song")
    chunk=[]
    notes=[]

print(dataset)
train_data = np.array(dataset)
np.save('preprocess_data',train_data,allow_pickle=True)