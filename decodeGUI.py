from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import numpy as np
import scipy.io.wavfile as wav
from matplotlib.pyplot import text
from scipy import signal


app = QApplication([])
window = QWidget()

window.resize(800, 600)
window.setWindowTitle("decodeGUI")
window.show()

selectFileButton = QPushButton("Select file", window)
selectFileButton.move(10, 10)
selectFileButton.resize(140, 40)
selectFileButton.show()

selectFileLabel = QLabel("", window)
selectFileLabel.move(170, 10)
selectFileLabel.resize(400, 40)
selectFileLabel.show()

processButton = QPushButton("Process file", window)
processButton.move(10, 60)
processButton.resize(140, 40)
processButton.show()

processLabel = QLabel("", window)
processLabel.move(170, 60)
processLabel.resize(400, 40)
processLabel.show()


inputFile = ""
def selectFile():
    global inputFile
    inputFile, check = QFileDialog.getOpenFileName(None, "Select File", ".", "WAV files (*.wav)")
    if check:
        selectFileLabel.setText(inputFile)
    return


resampleFactor = 1
start = 0
end = 0
def decode():
    if inputFile == "":
        updateProcessLabel("No file!")
        return
    
    updateProcessLabel("Started processing")

    sampleRate, data = wav.read(inputFile)
    updateProcessLabel("Loaded file")

    data = resample(data, sampleRate, resampleFactor)
    updateProcessLabel("Resampled data")

    data = crop(data, start, end, sampleRate)
    updateProcessLabel("Cropped data")

    amplitude = hilbert(data)
    updateProcessLabel("Got amplitude envelope")

    image = drawImage(amplitude, sampleRate)
    return

def resample(data, sampleRate, resampleFactor):
    data = data[::resampleFactor]
    sampleRate = int(sampleRate / resampleFactor)
    return data

def crop(data, start, end, sampleRate):
    if start > 0 or end > 0:
        startSample = int(start * sampleRate)
        endSample = int(end * sampleRate)
        data = data[startSample:endSample]
    return data

def hilbert(data):
    amplitude = np.abs(signal.hilbert(data))
    return amplitude

def drawImage(amplitude, sampleRate):

    return

def updateProcessLabel(str):
    processLabel.setText(str)
    return

selectFileButton.clicked.connect(selectFile)
processButton.clicked.connect(decode)

app.exec()
 