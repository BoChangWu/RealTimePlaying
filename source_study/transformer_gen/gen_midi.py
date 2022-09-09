from email import generator
import random
from mido import MidiFile, MidiTrack, Message
import numpy as np
from  .setting import seq_len
from .Transformer import TransformerGenerator

noise = np.random.normal(0,1,(1,seq_len))
predict = TransformerGenerator.predict(noise)
predict = predict*127

midler = MidiFile()
track = MidiTrack()
midler.tracks.append(track)

track.append(Message('program_change',program=2,time=0))

for x in range(seq_len):
    # 訓練的部分只有音符排列, 節奏跟控制都沒有訓練, 所以都是隨機生成
    on_interval = random.randint(0,127)
    off_interval = random.randint(0,127)
    change_interval = random.randint(0,127)
    change_value = random.randint(0,127)
    isControl = random.randint(0,1)
    track.append(Message('note_on',channel=1, note=int(predict[0][x]),velocity=64,time=on_interval))

    if isControl:
        track.append(Message('control_change',channel=1,control=64,value=change_value,time=change_interval))

    track.append(Message('note_off',channel = 1, note=int(predict[0][x]),velocity=64,time=off_interval))

    midler.save('MachineGen.mid')