import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import average
import scipy.io.wavfile as wav
import math
import time
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from scipy import signal

app = QApplication([])
window = QWidget()

window.resize(1800, 1100)
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

startLabel = QLabel("Start", window)
startLabel.move(10, 60)
startLabel.resize(100, 40)
startLabel.show()

startEntry = QLineEdit("0", window)
startEntry.setValidator(QDoubleValidator())
startEntry.setAlignment(Qt.AlignRight)
startEntry.move(130, 60)
startEntry.resize(80, 40)
startEntry.show()

endLabel = QLabel("End", window)
endLabel.move(10, 110)
endLabel.resize(100, 40)
endLabel.show()

endEntry = QLineEdit("0", window)
endEntry.setValidator(QDoubleValidator())
endEntry.setAlignment(Qt.AlignRight)
endEntry.move(130, 110)
endEntry.resize(80, 40)
endEntry.show()

resampleFactorLabel = QLabel("Resample", window)
resampleFactorLabel.move(10, 160)
resampleFactorLabel.resize(120, 40)
resampleFactorLabel.show()

resampleFactorEntry = QLineEdit("1", window)
resampleFactorEntry.setValidator(QIntValidator())
resampleFactorEntry.setAlignment(Qt.AlignRight)
resampleFactorEntry.move(130, 160)
resampleFactorEntry.resize(80, 40)
resampleFactorEntry.show()

shiftLabel = QLabel("Shift", window)
shiftLabel.move(10, 210)
shiftLabel.resize(100, 40)
shiftLabel.show()

shiftEntry = QLineEdit("0", window)
shiftEntry.setValidator(QIntValidator())
shiftEntry.setAlignment(Qt.AlignRight)
shiftEntry.move(130, 210)
shiftEntry.resize(80, 40)
shiftEntry.show()

brightnessLabel = QLabel("Brightness", window)
brightnessLabel.move(10, 260)
brightnessLabel.resize(100, 40)
brightnessLabel.show()

brightnessEntry = QLineEdit("1", window)
brightnessEntry.setValidator(QDoubleValidator())
brightnessEntry.setAlignment(Qt.AlignRight)
brightnessEntry.move(130, 260)
brightnessEntry.resize(80, 40)
brightnessEntry.show()

contrastLabel = QLabel("Contrast", window)
contrastLabel.move(10, 310)
contrastLabel.resize(100, 40)
contrastLabel.show()

contrastEntry = QLineEdit("1", window)
contrastEntry.setValidator(QDoubleValidator())
contrastEntry.setAlignment(Qt.AlignRight)
contrastEntry.move(130, 310)
contrastEntry.resize(80, 40)
contrastEntry.show()

processButton = QPushButton("Process file", window)
processButton.move(10, 1050)
processButton.resize(200, 40)
processButton.show()

processLabel = QLabel("", window)
processLabel.move(220, 1050)
processLabel.resize(600, 40)
processLabel.show()

saveButton = QPushButton("Save image", window)
saveButton.move(1590, 1050)
saveButton.resize(200, 40)
saveButton.show()

imageLabel = QLabel("", window)
imageLabel.move(220, 60)
imageLabel.resize(1360, 980)
imageLabel.show()


inputFile = ""
def selectFile():
    global inputFile
    inputFile, check = QFileDialog.getOpenFileName(None, "Select File", ".", "WAV files (*.wav)")
    if check:
        selectFileLabel.setText(inputFile)
    return


start = 0
end = 0
resampleFactor = 1
shift = 0
brightness = 1
contrast = 1
imageGenerated = False

def decode():
    global image, imageGenerated
    imageGenerated = False
    timeStart = time.time()

    if inputFile == "":
        update(processLabel, "No file!")
        return

    update(processLabel, "Loading file")
    sampleRate, data = wav.read(inputFile)

    update(processLabel, "Resampling data")
    data, sampleRate = resample(data, sampleRate, resampleFactor)

    update(processLabel, "Cropping data")
    data = crop(data, start, end, sampleRate)

    update(processLabel, "Generating amplitude envelope")
    amplitude = hilbert(data)

    update(processLabel, "Calculating average amplitude")
    averageAmplitude = getAverageAmplitude(amplitude)

    update(processLabel, "Generating image")
    image = generateImage(amplitude, sampleRate, averageAmplitude)

    update(processLabel, "Done (" + str(image.shape[1]) + "x" + str(image.shape[0]) + ", " + str(round(time.time() - timeStart, 2)) + "s)")
    display(image)

    imageGenerated = True
    return

def resample(data, sampleRate, resampleFactor):
    # for i in range(len(data)):
    #     if i % resampleFactor+1 == 0:
    #         for j in range(resampleFactor):
    #             data[i] += data[i+j]
    #         data[i] /= resampleFactor
            
    data = data[::resampleFactor]
    sampleRate = int(sampleRate / resampleFactor)
    return data, sampleRate

def crop(data, start, end, sampleRate):
    if end > start and (start > 0 or end > 0):
        startSample = int(start * sampleRate)
        endSample = int(end * sampleRate)
        data = data[startSample:endSample]
    return data

def hilbert(data):
    amplitude = np.abs(signal.hilbert(data))
    return amplitude

def getAverageAmplitude(amplitude):
    amplitude_sum = 0
    for i in range(amplitude.shape[0]):
        amplitude_sum += amplitude[i]
    averageAmplitude = float(amplitude_sum / amplitude.shape[0])
    return averageAmplitude

def generateImage(amplitude, sampleRate, averageAmplitude):
    width = int(0.5 * int(sampleRate + shift))
    height = int(amplitude.shape[0] / width)
    image = np.zeros((height, width, 3), dtype="uint8")
    
    x, y = 0, 0
    for p in range(amplitude.shape[0]):
        lum = int((amplitude[p] / (averageAmplitude * 2)) * 255 * brightness)
        offset = int((amplitude[p] - averageAmplitude) / averageAmplitude * 255)
        lum = int(lum + offset * math.log(contrast))
        if lum < 0:
            lum = 0
        if lum > 255:
            lum = 255
        image[y, x] = (lum, lum, lum)
        x += 1
        if x >= width:
            if y % 100 == 0:
                update(processLabel, "Generating image (" + str(y) + "/" + str(height) + ") * " + str(x))
            x = 0
            y += 1
            if y >= height:
                break
    return image

def display(image):
    h, w, _ = image.shape
    displayImage = QImage(image.data, w, h, 3 * w, QImage.Format_RGB888)
    displayPixMap = QPixmap(displayImage).scaled(imageLabel.size(), transformMode=Qt.TransformationMode.SmoothTransformation)
    imageLabel.setPixmap(displayPixMap)
    return

def save():
    if imageGenerated == True:
        path, check = QFileDialog.getSaveFileName(None, "Save Image", ".", "JPG file (*.jpg)")
        if check:
            plt.imsave(path, image)
    return

def update(object, str):
    object.setText(str)
    app.processEvents()
    return

def setStart(value):
    global start
    if value == "" or value == ".":
        value = 0
    start = float(value)
    return

def setEnd(value):
    global end
    if value == "" or value == ".":
        value = 0
    end = float(value)
    return

def setResampleFactor(value):
    global resampleFactor
    if value == "" or value == "-":
        value = 1
    resampleFactor = int(value)
    return

def setShift(value):
    global shift
    if value == "" or value == "-":
        value = 0
    shift = int(value)
    return

def setBrightness(value):
    global brightness
    if value == "" or value == ".":
        value = 1
    brightness = float(value)
    return

def setContrast(value):
    global contrast
    if value == "" or value == ".":
        value = 1
    contrast = float(value)
    return

selectFileButton.clicked.connect(selectFile)
processButton.clicked.connect(decode)
saveButton.clicked.connect(save)

startEntry.textChanged.connect(setStart)
endEntry.textChanged.connect(setEnd)
resampleFactorEntry.textChanged.connect(setResampleFactor)
shiftEntry.textChanged.connect(setShift)
brightnessEntry.textChanged.connect(setBrightness)
contrastEntry.textChanged.connect(setContrast)

app.exec()
 