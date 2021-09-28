from tkinter import *
from tkinter import Scale, filedialog, ttk

import numpy as np
import scipy.io.wavfile as wav
from matplotlib.pyplot import text
from scipy import signal

root = Tk()
root.title("decodeGUI")
root.geometry("800x600")

mainframe = ttk.Frame(root, padding="10 10 10 10")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)


inputFile = StringVar()
def loadFile():
    fd = filedialog.askopenfile(
        initialdir="C:/Users/Jonas/projects/personal/matura/py/", 
        title="Select WAV file", 
        filetypes=[("WAV files", ".wav")]
        )
    inputFile.set(fd.name)
    return

resampleFactor = IntVar()
resampleFactor.set(1)
start = IntVar()
end = IntVar()
def decode():
    sampleRate, data = wav.read(inputFile.get())
    data = resample(data, sampleRate, resampleFactor.get())
    data = crop(data, start, end, sampleRate)
    amplitude = hilbert(data)
    return

def resample(data, sampleRate, resampleFactor):
    data = data[::resampleFactor]
    sampleRate = int(sampleRate / resampleFactor)
    print("resampled data")
    return data

def crop(data, start, end, sampleRate):
    if start.get() > 0 or end.get() > 0:
        startSample = int(start.get() * sampleRate)
        endSample = int(end.get() * sampleRate)
        data = data[startSample:endSample]
    return data

def hilbert(data):
    amplitude = np.abs(signal.hilbert(data))
    print("get amplitude envelope")
    return amplitude

ttk.Button(mainframe, text="Load File", command=loadFile).grid(column=0, row=0, sticky=W)
ttk.Label(mainframe, textvariable=inputFile).grid(column=1, row=0, sticky=W)

ttk.Button(mainframe, text="Decode File", command=decode).grid(column=0, row=1, sticky=W)

ttk.Spinbox(mainframe, from_=0, to=9999, textvariable=start).grid(column=0, row=2, sticky=W)
ttk.Spinbox(mainframe, from_=0, to=9999, textvariable=end).grid(column=0, row=3, sticky=W)

Scale(mainframe, variable=resampleFactor, from_=1, to=16, resolution=1, orient=HORIZONTAL).grid(column=0, row=4, sticky=W)

root.mainloop()
