#!/usr/bin/python

from sys import byteorder
from array import array
from struct import pack

import pyaudio
import wave
import numpy as np

import time

THRESHOLD = 500
#CHUNK_SIZE = 1000
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 8000
RECORD_SECONDS = 2
NUM_FILES = 200
DATA_PREPEND = .5

class RingBuffer():
    "A 1D ring buffer using numpy arrays"
    def __init__(self, length):
        self.data = np.zeros(length, dtype=np.int16)
        self.index = 0
	
    def extend(self, x):
        "adds array x to ring buffer"
        x_index = (self.index + np.arange(x.size)) % self.data.size
        self.data[x_index] = x
        self.index = x_index[-1] + 1

    def get(self):
        "Returns the first-in-first-out data in the ring buffer"
        idx = (self.index + np.arange(self.data.size)) %self.data.size
        return self.data[idx]

def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    #maxval = max(snd_data)
    #print( "Max sound data = %d" % (maxval))
    return max(snd_data) < THRESHOLD

def record():
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.5 seconds of 
    blank sound to make sure VLC et al can play 
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    num_samples_taken = 0
    snd_started = False
    recorded_time = 0.0
    prepend_samples = 0
    prepend_done = False

    #r = array('h')
    r = RingBuffer(RECORD_SECONDS * RATE)

    while 1:
	snd_data = stream.read(CHUNK_SIZE)
	audio_data = np.fromstring(snd_data, dtype=np.int16)
        r.extend(audio_data)
        
	if not prepend_done:
	    prepend_samples += 1
	    #print("Recorded time = %f" % recorded_time)
            recorded_time += float(CHUNK_SIZE) / float(RATE)
        
	if recorded_time >= DATA_PREPEND:
	    #print("Prepend done")
            prepend_done = True

	if snd_started and prepend_done:
	    num_samples_taken += 1
	    if num_samples_taken > ((RECORD_SECONDS - recorded_time) * RATE / CHUNK_SIZE):
		break;
	
        if not snd_started:
	    if not is_silent(audio_data):
                print("Starting recording")
                snd_started = True

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    return sample_width, r.get()

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    #data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

if __name__ == '__main__':
    for i in range(0,NUM_FILES):
        timestr = time.strftime("%Y%m%d-%H%M%S")
	file_name = timestr + "file_" + `i` + ".wav"
        record_to_file(file_name)
	print("done - result written to %s" % (file_name))
