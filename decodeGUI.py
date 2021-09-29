import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from scipy import signal

app = QApplication([])
window = QWidget()

window.resize(800, 600)
window.setWindowTitle("decodeGUI")
window.show()

selectFileButton = QPushButton("Select file", window)
selectFileButton.move(10, 10)
selectFileButton.resize(200, 40)
selectFileButton.show()

selectFileLabel = QLabel("", window)
selectFileLabel.move(220, 10)
selectFileLabel.resize(600, 40)
selectFileLabel.show()

processButton = QPushButton("Process file", window)
processButton.move(10, 550)
processButton.resize(200, 40)
processButton.show()

processLabel = QLabel("", window)
processLabel.move(220, 550)
processLabel.resize(600, 40)
processLabel.show()

startLabel = QLabel("Start", window)
startLabel.move(10, 60)
startLabel.resize(100, 40)
startLabel.show()

startEntry = QLineEdit("", window)
startEntry.setValidator(QIntValidator())
startEntry.setAlignment(Qt.AlignRight)
startEntry.move(130, 60)
startEntry.resize(80, 40)
startEntry.show()

endLabel = QLabel("End", window)
endLabel.move(10, 110)
endLabel.resize(100, 40)
endLabel.show()

endEntry = QLineEdit("", window)
endEntry.setValidator(QIntValidator())
endEntry.setAlignment(Qt.AlignRight)
endEntry.move(130, 110)
endEntry.resize(80, 40)
endEntry.show()

resampleFactorLabel = QLabel("Resample", window)
resampleFactorLabel.move(10, 160)
resampleFactorLabel.resize(100, 40)
resampleFactorLabel.show()

resampleFactorValueLabel = QLabel("1", window)
resampleFactorValueLabel.move(180, 160)
resampleFactorValueLabel.resize(40, 40)
resampleFactorValueLabel.show()

resampleFactorSlider = QSlider(Qt.Orientation.Horizontal, window)
resampleFactorSlider.setMinimum(1)
resampleFactorSlider.setMaximum(16)
resampleFactorSlider.setTickInterval(1)
resampleFactorSlider.setTickPosition(1)
resampleFactorSlider.move(10, 210)
resampleFactorSlider.resize(200, 40)
resampleFactorSlider.show()


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
        update(processLabel, "No file!")
        return
    
    update(processLabel, "Start processing")

    update(processLabel, "Loading file")
    sampleRate, data = wav.read(inputFile)

    update(processLabel, "Resampling data")
    data = resample(data, sampleRate, resampleFactor)

    update(processLabel, "Cropping data")
    data = crop(data, start, end, sampleRate)

    update(processLabel, "Generating amplitude envelope")
    amplitude = hilbert(data)

    update(processLabel, "Generating image")
    image = generateImage(amplitude, sampleRate)
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

def generateImage(amplitude, sampleRate):

    return

def update(object, str):
    object.setText(str)
    app.processEvents()
    return

def setResampleFactor(value):
    global resampleFactor
    resampleFactor = value
    update(resampleFactorValueLabel, str(value))

def setStart(value):
    global start
    if value == "":
        value = 0
    start = int(value)

def setEnd(value):
    global end
    if value == "":
        value = 0
    end = int(value)

selectFileButton.clicked.connect(selectFile)
processButton.clicked.connect(decode)
resampleFactorSlider.valueChanged.connect(setResampleFactor)

startEntry.textChanged.connect(setStart)
endEntry.textChanged.connect(setEnd)

app.exec()
 