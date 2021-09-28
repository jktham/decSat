from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import Scale
from matplotlib.pyplot import text

import scipy.io.wavfile as wav

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

def resample():
    return

resampleFactor = IntVar()
resampleFactor.set(1)
def decode():
    sampleRate, data = wav.read(inputFile.get())
    data = resample(data, sampleRate, resampleFactor.get())
    return

def resample(data, sampleRate, resampleFactor):
    data = data[::resampleFactor]
    sampleRate = int(sampleRate / resampleFactor)
    print("resampled data")
    return data

ttk.Button(mainframe, text="Load File", command=loadFile).grid(column=0, row=0, sticky=W)
ttk.Label(mainframe, textvariable=inputFile).grid(column=1, row=0, sticky=W)

ttk.Button(mainframe, text="Decode File", command=decode).grid(column=0, row=1, sticky=W)

Scale(mainframe, variable=resampleFactor, from_=1, to=16, resolution=1, orient=HORIZONTAL).grid(column=0, row=2, sticky=W)

root.mainloop()