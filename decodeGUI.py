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
from skimage import color, transform

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

shiftLabel = QLabel("Shift", window)
shiftLabel.move(10, 160)
shiftLabel.resize(100, 40)
shiftLabel.show()

shiftEntry = QLineEdit("0", window)
shiftEntry.setValidator(QIntValidator())
shiftEntry.setAlignment(Qt.AlignRight)
shiftEntry.move(130, 160)
shiftEntry.resize(80, 40)
shiftEntry.show()

resampleFactorLabel = QLabel("Resample", window)
resampleFactorLabel.move(10, 210)
resampleFactorLabel.resize(120, 40)
resampleFactorLabel.show()

resampleFactorEntry = QLineEdit("1", window)
resampleFactorEntry.setValidator(QIntValidator())
resampleFactorEntry.setAlignment(Qt.AlignRight)
resampleFactorEntry.move(130, 210)
resampleFactorEntry.resize(80, 40)
resampleFactorEntry.show()

resampleFactorSlider = QSlider(window)
resampleFactorSlider.setOrientation(Qt.Orientation.Horizontal)
resampleFactorSlider.setTickInterval(1)
resampleFactorSlider.setTickPosition(3)
resampleFactorSlider.setValue(0)
resampleFactorSlider.setMinimum(0)
resampleFactorSlider.setMaximum(4)
resampleFactorSlider.move(10, 260)
resampleFactorSlider.resize(200, 40)
resampleFactorSlider.show()

brightnessLabel = QLabel("Brightness", window)
brightnessLabel.move(10, 310)
brightnessLabel.resize(100, 40)
brightnessLabel.show()

brightnessEntry = QLineEdit("1.0", window)
brightnessEntry.setValidator(QDoubleValidator())
brightnessEntry.setAlignment(Qt.AlignRight)
brightnessEntry.move(130, 310)
brightnessEntry.resize(80, 40)
brightnessEntry.show()

brightnessSlider = QSlider(window)
brightnessSlider.setOrientation(Qt.Orientation.Horizontal)
brightnessSlider.setTickInterval(1)
brightnessSlider.setTickPosition(3)
brightnessSlider.setValue(10)
brightnessSlider.setMinimum(5)
brightnessSlider.setMaximum(15)
brightnessSlider.move(10, 360)
brightnessSlider.resize(200, 40)
brightnessSlider.show()

contrastLabel = QLabel("Contrast", window)
contrastLabel.move(10, 410)
contrastLabel.resize(100, 40)
contrastLabel.show()

contrastEntry = QLineEdit("1.0", window)
contrastEntry.setValidator(QDoubleValidator())
contrastEntry.setAlignment(Qt.AlignRight)
contrastEntry.move(130, 410)
contrastEntry.resize(80, 40)
contrastEntry.show()

contrastSlider = QSlider(window)
contrastSlider.setOrientation(Qt.Orientation.Horizontal)
contrastSlider.setTickInterval(1)
contrastSlider.setTickPosition(3)
contrastSlider.setValue(10)
contrastSlider.setMinimum(5)
contrastSlider.setMaximum(15)
contrastSlider.move(10, 460)
contrastSlider.resize(200, 40)
contrastSlider.show()

filterLabel = QLabel("Filter", window)
filterLabel.move(10, 510)
filterLabel.resize(100, 40)
filterLabel.show()

filterCheckbox = QCheckBox(window)
filterCheckbox.setChecked(True)
filterCheckbox.move(160, 510)
filterCheckbox.resize(40, 40)
filterCheckbox.show()

processButton = QPushButton("Process file", window)
processButton.move(10, 1050)
processButton.resize(200, 40)
processButton.show()

processLabel = QLabel("", window)
processLabel.move(220, 1050)
processLabel.resize(600, 40)
processLabel.show()

clearButton = QPushButton("Clear image", window)
clearButton.move(1590, 10)
clearButton.resize(200, 40)
clearButton.show()

plotWavButton = QPushButton("Plot wav", window)
plotWavButton.move(1590, 60)
plotWavButton.resize(200, 40)
plotWavButton.show()

plotImageButton = QPushButton("Plot image", window)
plotImageButton.move(1590, 110)
plotImageButton.resize(200, 40)
plotImageButton.show()

plotWavFourierButton = QPushButton("Plot wav fourier", window)
plotWavFourierButton.move(1590, 160)
plotWavFourierButton.resize(200, 40)
plotWavFourierButton.show()

plotImageFourierPreButton = QPushButton("Plot image fourier (pre)", window)
plotImageFourierPreButton.move(1590, 210)
plotImageFourierPreButton.resize(200, 40)
plotImageFourierPreButton.show()

plotImageFourierPostButton = QPushButton("Plot image fourier (post)", window)
plotImageFourierPostButton.move(1590, 260)
plotImageFourierPostButton.resize(200, 40)
plotImageFourierPostButton.show()

saveButton = QPushButton("Save image", window)
saveButton.move(1590, 1050)
saveButton.resize(200, 40)
saveButton.show()

imageDisplayLabel = QLabel("", window)
imageDisplayLabel.move(220, 60)
imageDisplayLabel.resize(1360, 980)
imageDisplayLabel.show()

aspectRatioCheckbox = QCheckBox(window)
aspectRatioCheckbox.setChecked(True)
aspectRatioCheckbox.move(1590, 1000)
aspectRatioCheckbox.resize(40, 40)
aspectRatioCheckbox.show()

aspectRatioEntryLabel = QLabel("Aspect ratio", window)
aspectRatioEntryLabel.move(1620, 1000)
aspectRatioEntryLabel.resize(200, 40)
aspectRatioEntryLabel.show()

aspectRatioEntry = QLineEdit("1.4", window)
aspectRatioEntry.setValidator(QDoubleValidator())
aspectRatioEntry.setAlignment(Qt.AlignRight)
aspectRatioEntry.move(1730, 1000)
aspectRatioEntry.resize(60, 40)
aspectRatioEntry.show()


start = 0
end = 0
resampleFactor = 1
shift = 0
brightness = 1
contrast = 1
processingDone = False
inputFile = ""
aspectRatio = 1.4

def decode():
    global data, originalSampleRate, sampleRate, amplitude, averageAmplitude, image, processingDone
    processingDone = False
    timeStart = time.time()

    if inputFile == "":
        update(processLabel, "No file!")
        return

    update(processLabel, "Loading file")
    originalSampleRate, data = wav.read(inputFile)

    update(processLabel, "Resampling data")
    data, sampleRate = resample(data, originalSampleRate, resampleFactor)

    update(processLabel, "Cropping data")
    data = crop(data, start, end, sampleRate)

    update(processLabel, "Generating amplitude envelope")
    amplitude = hilbert(data)

    update(processLabel, "Calculating average amplitude")
    averageAmplitude = getAverageAmplitude(amplitude)

    update(processLabel, "Generating image")
    image = generateImage(amplitude, sampleRate, averageAmplitude)

    update(processLabel, "Filtering image")
    image = filter(image)

    update(processLabel, "Done (" + str(image.shape[1]) + "x" + str(image.shape[0]) + ", " + str(round(time.time() - timeStart, 2)) + "s)")
    display(image)

    processingDone = True
    return

def selectFile():
    global inputFile
    inputFile, check = QFileDialog.getOpenFileName(None, "Select File", ".", "WAV files (*.wav)")
    if check:
        update(selectFileLabel, inputFile)
    else:
        update(selectFileLabel, "")
    return

def resample(data, sampleRate, resampleFactor):
    data = data[::resampleFactor]
    sampleRate = int(sampleRate / resampleFactor)
    return data, sampleRate

def crop(data, start, end, sampleRate):
    if end > start and (start > 0 or end > 0):
        startSample = int(start * sampleRate)
        endSample = int(end * sampleRate)
        data = data[startSample:endSample]
    return data

def filter(image):
    global imageFourierPre
    if filterCheckbox.isChecked():
        for c in range(image.shape[2]):
            imageFourierPre = np.fft.fftshift(np.fft.fft2(image[:, :, c]))
            for i in range(imageFourierPre.shape[0]):
                for j in range(imageFourierPre.shape[1]):
                    if 1250 < j < 1750:
                        imageFourierPre[i, j] = 1
                    if 3750 < j < 4250:
                        imageFourierPre[i, j] = 1
            image[:, :, c] = np.real(np.fft.ifft2(np.fft.ifftshift(imageFourierPre)))
    return image

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
    global width, height
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
            if y % 10 == 0:
                update(processLabel, "Generating image (" + str(y) + "/" + str(height) + ") * " + str(x))
            x = 0
            y += 1
            if y >= height:
                break
    return image

def display(image):
    h, w, _ = image.shape
    displayImage = QImage(image.data, w, h, 3 * w, QImage.Format_RGB888)
    displayPixMap = QPixmap(displayImage).scaled(imageDisplayLabel.size(), transformMode=Qt.TransformationMode.SmoothTransformation)
    imageDisplayLabel.setPixmap(displayPixMap)
    return

def save():
    global image
    if processingDone:
        path, check = QFileDialog.getSaveFileName(None, "Save Image", ".", "PNGfile (*.png)")
        if check:
            if aspectRatioCheckbox.isChecked():
                aspectRatioImage = transform.resize(image, (height, int(height * aspectRatio)))
                plt.imsave(path, aspectRatioImage)
            else:
                plt.imsave(path, image)
    return

def clear():
    global processingDone, image
    if processingDone:
        processingDone = False
        imageDisplayLabel.setPixmap(QPixmap())
        processLabel.setText("")
    return

def update(object, str):
    object.setText(str)
    app.processEvents()
    return

def setStart(value):
    global start
    if value == "" or value == "." or value == "-":
        value = 0
    start = float(value)
    return

def setEnd(value):
    global end
    if value == "" or value == "." or value == "-":
        value = 0
    end = float(value)
    return

def setShift(value):
    global shift
    if value == "" or value == "-":
        value = 0
    shift = int(value)
    return

def setResampleFactor(value):
    global resampleFactor
    if value == "" or value == "-":
        value = 1
    resampleFactor = int(value)
    return

def setResampleFactorSlider(value):
    global resampleFactor
    resampleFactor = 2 ** int(value)
    resampleFactorEntry.setText(str(2 ** int(value)))
    return

def setBrightness(value):
    global brightness
    if value == "" or value == "." or value == "-":
        value = 1
    brightness = float(value)
    return

def setBrightnessSlider(value):
    global brightness
    brightness = float(value/1000)
    brightnessEntry.setText(str(int(value)/10))
    return

def setContrast(value):
    global contrast
    if value == "" or value == "." or value == "-":
        value = 1
    contrast = float(value)
    return

def setContrastSlider(value):
    global contrast
    contrast = float(int(value)/1000)
    contrastEntry.setText(str(int(value)/10))
    return

def setAspectRatio(value):
    global aspectRatio
    if value == "" or value == "." or value == "-":
        value = 1
    aspectRatio = float(value)
    return

def plotImage():
    if processingDone:
        plt.ion()
        plt.figure(2, figsize=(24, 16))
        plt.imshow(image, aspect=image.shape[1] / image.shape[0] * 0.8)
        plt.title(
            inputFile + ", " + 
            str(originalSampleRate) + "Hz, (" + 
            str(sampleRate) + "), " + 
            str(start) + "-" + 
            str(end) + "s, " + 
            str(image.shape[1]) + "x" + 
            str(image.shape[0]) + ", " + 
            str(brightness) + ", " + 
            str(contrast)
        )
        plt.show()
    return

def plotWav():
    if processingDone:
        plt.ion()
        plt.figure(0, figsize=(24, 8))
        plt.plot(data)
        plt.plot(amplitude)
        plt.axhline(y=averageAmplitude, color='r', linestyle='-')
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.title(
            inputFile + ", " + 
            str(originalSampleRate) + "Hz, (" + 
            str(sampleRate) + "), " + 
            str(start) + "-" + 
            str(end) + "s"
        )
        plt.show()
    return

def plotImageFourierPre():
    global imageFourierPre
    if processingDone:
        if filterCheckbox.isChecked():
            plt.ion()
            plt.figure(3, figsize=(8, 8))
            for c in range(image.shape[2]):
                plt.imshow(np.log(abs(imageFourierPre)), cmap="gray", aspect=image.shape[1] / image.shape[0] * 0.8)
            plt.title("Image fourier")
            plt.show()
    return

def plotImageFourierPost():
    if processingDone:
        plt.ion()
        plt.figure(3, figsize=(8, 8))
        for c in range(image.shape[2]):
            imageFourierPost = np.fft.fftshift(np.fft.fft2(image[:, :, c]))
            plt.imshow(np.log(abs(imageFourierPost)), cmap="gray", aspect=image.shape[1] / image.shape[0] * 0.8)
        plt.title("Image fourier")
        plt.show()
    return

def plotWavFourier():
    if processingDone:
        plt.ion()
        plt.figure(1, figsize=(8, 8))
        plt.plot(np.real(np.fft.fftshift(np.fft.fft(data))))
        plt.title("Wav fourier")
        plt.show()
    return

selectFileButton.clicked.connect(selectFile)
processButton.clicked.connect(decode)
saveButton.clicked.connect(save)
clearButton.clicked.connect(clear)
plotImageButton.clicked.connect(plotImage)
plotWavButton.clicked.connect(plotWav)
plotImageFourierPreButton.clicked.connect(plotImageFourierPre)
plotImageFourierPostButton.clicked.connect(plotImageFourierPost)
plotWavFourierButton.clicked.connect(plotWavFourier)

startEntry.textChanged.connect(setStart)
endEntry.textChanged.connect(setEnd)
shiftEntry.textChanged.connect(setShift)
resampleFactorEntry.textChanged.connect(setResampleFactor)
brightnessEntry.textChanged.connect(setBrightness)
contrastEntry.textChanged.connect(setContrast)
aspectRatioEntry.textChanged.connect(setAspectRatio)

brightnessSlider.valueChanged.connect(setBrightnessSlider)
contrastSlider.valueChanged.connect(setContrastSlider)
resampleFactorSlider.valueChanged.connect(setResampleFactorSlider)

app.exec_()