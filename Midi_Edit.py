import random
from mido import MidiFile , MidiTrack, Message


class Midi_Encode():
    def __init__(self,):
        self.midler = MidiFile()
        self.track = MidiTrack()

        self.midler.tracks.append(self.track)
        self.track.append(Message('program_change',program=2,time=0))

    def make_file(self,params,notes_nums=256,name='Midi_Sample.mid'):
        for x in range(notes_nums):
            on_interval = random.randint(50,127)
            off_interval = random.randint(0,127)
            change_interval = random.randint(0,127)
            change_value = random.randint(0,127)
            isControl = random.randint(0,1)
    
        self.track.append(Message('note_on', channel=1, note=int(params[0][x][0]), velocity=64, time=on_interval))

        if isControl:
            self.track.append(Message('control_change', channel=1, control=64, value=change_value, time= change_interval))

        self.track.append(Message('note_off', channel=1, note=int(params[0][x][0]), velocity=64, time=off_interval))

        self.midler.save(name)