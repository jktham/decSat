import math
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from scipy import signal
from skimage import color, transform

# --- UI setup ---

app = QApplication([])
window = QWidget()

window.resize(1800, 1100)
window.setWindowTitle("decodeGUI")
window.show()

select_file_button = QPushButton("Select file", window)
select_file_button.move(10, 10)
select_file_button.resize(200, 40)
select_file_button.show()

select_file_label = QLabel("", window)
select_file_label.move(220, 10)
select_file_label.resize(800, 40)
select_file_label.show()

start_label = QLabel("Start", window)
start_label.move(10, 60)
start_label.resize(100, 40)
start_label.show()

start_entry = QLineEdit("0", window)
start_entry.setValidator(QDoubleValidator())
start_entry.setAlignment(Qt.AlignRight)
start_entry.move(130, 60)
start_entry.resize(80, 40)
start_entry.show()

end_label = QLabel("End", window)
end_label.move(10, 110)
end_label.resize(100, 40)
end_label.show()

end_entry = QLineEdit("0", window)
end_entry.setValidator(QDoubleValidator())
end_entry.setAlignment(Qt.AlignRight)
end_entry.move(130, 110)
end_entry.resize(80, 40)
end_entry.show()

shift_label = QLabel("Shift", window)
shift_label.move(10, 160)
shift_label.resize(100, 40)
shift_label.show()

shift_entry = QLineEdit("0", window)
shift_entry.setValidator(QIntValidator())
shift_entry.setAlignment(Qt.AlignRight)
shift_entry.move(130, 160)
shift_entry.resize(80, 40)
shift_entry.show()

resample_factor_label = QLabel("Resample", window)
resample_factor_label.move(10, 210)
resample_factor_label.resize(120, 40)
resample_factor_label.show()

resample_factor_entry = QLineEdit("1", window)
resample_factor_entry.setValidator(QIntValidator())
resample_factor_entry.setAlignment(Qt.AlignRight)
resample_factor_entry.move(130, 210)
resample_factor_entry.resize(80, 40)
resample_factor_entry.show()

resample_factor_slider = QSlider(window)
resample_factor_slider.setOrientation(Qt.Orientation.Horizontal)
resample_factor_slider.setTickInterval(1)
resample_factor_slider.setTickPosition(3)
resample_factor_slider.setValue(0)
resample_factor_slider.setMinimum(0)
resample_factor_slider.setMaximum(4)
resample_factor_slider.move(10, 260)
resample_factor_slider.resize(200, 40)
resample_factor_slider.show()

brightness_label = QLabel("Brightness", window)
brightness_label.move(10, 310)
brightness_label.resize(100, 40)
brightness_label.show()

brightness_entry = QLineEdit("1.0", window)
brightness_entry.setValidator(QDoubleValidator())
brightness_entry.setAlignment(Qt.AlignRight)
brightness_entry.move(130, 310)
brightness_entry.resize(80, 40)
brightness_entry.show()

brightness_slider = QSlider(window)
brightness_slider.setOrientation(Qt.Orientation.Horizontal)
brightness_slider.setTickInterval(1)
brightness_slider.setTickPosition(3)
brightness_slider.setValue(10)
brightness_slider.setMinimum(5)
brightness_slider.setMaximum(15)
brightness_slider.move(10, 360)
brightness_slider.resize(200, 40)
brightness_slider.show()

contrast_label = QLabel("Contrast", window)
contrast_label.move(10, 410)
contrast_label.resize(100, 40)
contrast_label.show()

contrast_entry = QLineEdit("1.0", window)
contrast_entry.setValidator(QDoubleValidator())
contrast_entry.setAlignment(Qt.AlignRight)
contrast_entry.move(130, 410)
contrast_entry.resize(80, 40)
contrast_entry.show()

contrast_slider = QSlider(window)
contrast_slider.setOrientation(Qt.Orientation.Horizontal)
contrast_slider.setTickInterval(1)
contrast_slider.setTickPosition(3)
contrast_slider.setValue(10)
contrast_slider.setMinimum(5)
contrast_slider.setMaximum(15)
contrast_slider.move(10, 460)
contrast_slider.resize(200, 40)
contrast_slider.show()

offset_label = QLabel("Offset", window)
offset_label.move(10, 510)
offset_label.resize(100, 40)
offset_label.show()

offset_entry = QLineEdit("0.0", window)
offset_entry.setValidator(QDoubleValidator())
offset_entry.setAlignment(Qt.AlignRight)
offset_entry.move(130, 510)
offset_entry.resize(80, 40)
offset_entry.show()

offset_slider = QSlider(window)
offset_slider.setOrientation(Qt.Orientation.Horizontal)
offset_slider.setTickInterval(1)
offset_slider.setTickPosition(3)
offset_slider.setValue(0)
offset_slider.setMinimum(-10)
offset_slider.setMaximum(10)
offset_slider.move(10, 560)
offset_slider.resize(200, 40)
offset_slider.show()

high_pass_filter_label = QLabel("High pass filter", window)
high_pass_filter_label.move(10, 610)
high_pass_filter_label.resize(160, 40)
high_pass_filter_label.show()

high_pass_filter_checkbox = QCheckBox(window)
high_pass_filter_checkbox.setChecked(True)
high_pass_filter_checkbox.move(180, 610)
high_pass_filter_checkbox.resize(40, 40)
high_pass_filter_checkbox.show()

fourier_filter_label = QLabel("Fourier filter", window)
fourier_filter_label.move(10, 660)
fourier_filter_label.resize(120, 40)
fourier_filter_label.show()

fourier_filter_checkbox = QCheckBox(window)
fourier_filter_checkbox.setChecked(False)
fourier_filter_checkbox.move(180, 660)
fourier_filter_checkbox.resize(40, 40)
fourier_filter_checkbox.show()

resync_label = QLabel("Resync", window)
resync_label.move(10, 710)
resync_label.resize(120, 40)
resync_label.show()

resync_checkbox = QCheckBox(window)
resync_checkbox.setChecked(False)
resync_checkbox.move(180, 710)
resync_checkbox.resize(40, 40)
resync_checkbox.show()


process_button = QPushButton("Process file", window)
process_button.move(10, 1050)
process_button.resize(200, 40)
process_button.show()

process_label = QLabel("", window)
process_label.move(220, 1050)
process_label.resize(600, 40)
process_label.show()


plot_wav_button = QPushButton("Plot wav", window)
plot_wav_button.move(1590, 10)
plot_wav_button.resize(200, 40)
plot_wav_button.show()

plot_wav_fourier_button = QPushButton("Plot wav fourier", window)
plot_wav_fourier_button.move(1590, 60)
plot_wav_fourier_button.resize(200, 40)
plot_wav_fourier_button.show()

plot_spectrogram_button = QPushButton("Plot spectrogram", window)
plot_spectrogram_button.move(1590, 110)
plot_spectrogram_button.resize(200, 40)
plot_spectrogram_button.show()

plot_image_button = QPushButton("Plot image", window)
plot_image_button.move(1590, 160)
plot_image_button.resize(200, 40)
plot_image_button.show()

plot_image_fourier_pre_button = QPushButton("Plot fourier (pre)", window)
plot_image_fourier_pre_button.move(1590, 210)
plot_image_fourier_pre_button.resize(200, 40)
plot_image_fourier_pre_button.show()

plot_image_fourier_post_button = QPushButton("Plot fourier (post)", window)
plot_image_fourier_post_button.move(1590, 260)
plot_image_fourier_post_button.resize(200, 40)
plot_image_fourier_post_button.show()


aspect_ratio_checkbox = QCheckBox(window)
aspect_ratio_checkbox.setChecked(True)
aspect_ratio_checkbox.move(1590, 1000)
aspect_ratio_checkbox.resize(40, 40)
aspect_ratio_checkbox.show()

aspect_ratio_entryLabel = QLabel("Aspect", window)
aspect_ratio_entryLabel.move(1630, 1000)
aspect_ratio_entryLabel.resize(200, 40)
aspect_ratio_entryLabel.show()

aspect_ratio_entry = QLineEdit("1.4", window)
aspect_ratio_entry.setValidator(QDoubleValidator())
aspect_ratio_entry.setAlignment(Qt.AlignRight)
aspect_ratio_entry.move(1710, 1000)
aspect_ratio_entry.resize(80, 40)
aspect_ratio_entry.show()

save_button = QPushButton("Save image", window)
save_button.move(1590, 1050)
save_button.resize(200, 40)
save_button.show()


image_label = QLabel("", window)
image_label.move(220, 60)
image_label.resize(1360, 980)
image_label.show()

info_label = QLabel("", window)
info_label.move(400, 1050)
info_label.resize(800, 40)
info_label.show()

# --- Processing ---

input_file = ""
start = 0
end = 0
shift = 0
resample_factor = 1
resample_rate = 4160
brightness = 1
contrast = 1
offset = 0
aspect_ratio = 1.4

processing_done = False
filtering_done = False

def decode():
    global data, original_sample_rate, sample_rate, amplitude, average_amplitude, image, processing_done, filtering_done
    processing_done = False
    filtering_done = False
    time_start = time.time()

    if input_file == "":
        updateText(process_label, "No file!")
        return

    updateText(info_label, "")

    updateText(process_label, "Loading file")
    original_sample_rate, data = wav.read(input_file)

    updateText(process_label, "Resampling data")
    data, sample_rate = resample(data, original_sample_rate, resample_factor, resample_rate)

    updateText(process_label, "Cropping data")
    data = crop(data, start, end, sample_rate)

    updateText(process_label, "Filtering data (High pass)")
    data = highPassFilter(data)

    updateText(process_label, "Generating amplitude envelope")
    amplitude = envelope(data)

    updateText(process_label, "Calculating average amplitude")
    average_amplitude = getAverageAmplitude(amplitude)

    updateText(process_label, "Generating image")
    image = generateImage(amplitude, sample_rate, average_amplitude)

    updateText(process_label, "Offsetting image")
    image = applyOffset(image, offset)

    updateText(process_label, "Resyncing image")
    image = resync(image)

    updateText(process_label, "Filtering image (Fourier)")
    image = fourierFilter(image)

    updateText(process_label, f"Done ({str(round(time.time() - time_start, 2))}s)")
    processing_done = True
    displayImage(image)
    displayInfo()
    return

def selectFile():
    global input_file
    input_file, check = QFileDialog.getOpenFileName(None, "Select File", "C:/Users/Jonas/projects/personal/matura/py/in", "WAV files (*.wav)")
    if check:
        updateText(select_file_label, input_file)
    else:
        updateText(select_file_label, "")
    return

def resample(data, sample_rate, resample_factor, resample_rate):
    # data = signal.resample(data, int(data.shape[0] / sample_rate) * 20800)
    # data = signal.decimate(data, 5)
    # sample_rate = resample_rate
    data = data[::resample_factor]
    sample_rate = int(sample_rate / resample_factor)
    return data, sample_rate

def crop(data, start, end, sample_rate):
    if end > start and (start > 0 or end > 0):
        startSample = int(start * sample_rate)
        endSample = int(end * sample_rate)
        data = data[startSample:endSample]
    return data

def highPassFilter(data):
    passes = 1
    if high_pass_filter_checkbox.isChecked():
        hpf = signal.firwin(101, 1200, fs=sample_rate, pass_zero=False)
        for i in range(passes):
            data = signal.lfilter(hpf, [1.0], data)
    return data

def envelope(data):
    amplitude = np.abs(signal.hilbert(data))
    return amplitude

def getAverageAmplitude(amplitude):
    amplitude_sum = 0
    for i in range(amplitude.shape[0]):
        amplitude_sum += amplitude[i]
    average_amplitude = float(amplitude_sum / amplitude.shape[0])
    return average_amplitude

def generateImage(amplitude, sample_rate, average_amplitude):
    global width, height
    width = int(0.5 * int(sample_rate + shift))
    height = int(amplitude.shape[0] / width)
    image = np.zeros((height, width, 3), dtype="uint8")
    
    x, y = 0, 0
    for p in range(amplitude.shape[0]):
        lum = int((amplitude[p] / (average_amplitude * 2)) * 255 * brightness)
        offset = int((amplitude[p] - average_amplitude) / average_amplitude * 255)
        lum = int(lum + offset * math.log(contrast))
        if lum < 0:
            lum = 0
        if lum > 255:
            lum = 255
        image[y, x] = (lum, lum, lum)
        x += 1
        if x >= width:
            if y % 10 == 0:
                updateText(process_label, f"Generating image ({str(y)}/{str(height)}) * {str(x)}")
            x = 0
            y += 1
            if y >= height:
                break
    return image

def applyOffset(image, offset):
    for y in range(height):
        image[y] = np.roll(image[y], int(offset*width), axis=0)
    return image

def resync(image):
    if resync_checkbox.isChecked():
        sync_pos = np.zeros(height)
        start_pos = 0
        count_pos = 0

        for y in range(height):
            if sync_pos[y-1] > 0:
                start_pos = abs(int(sync_pos[y-1] - 100))
            for x in range(start_pos, width+start_pos):
                if x >= width:
                    x = x-width
                dark_sum = 0
                bright_sum = 0

                for k in range(130):
                    if x+k < width:
                        dark_sum += image[y, x+k, 0]
                    else:
                        dark_sum += image[y, x-k, 0]
                    if x+k+width/2 < width:
                        bright_sum += image[y, int(x+k+width/2), 0]
                    else:
                        bright_sum += image[y, int(x+k-width/2), 0]

                if (dark_sum < 130 * 40 and bright_sum > 130 * 240): # or (dark_sum > 140 * 220 and bright_sum > 140 * 220) or (dark_sum < 140 * 50 and bright_sum < 140 * 50):
                    sync_pos[y] = x
                    count_pos += 1
                    updateText(process_label, f"Resyncing image ({y}/{int(height)}, s {int(sync_pos[y])}, {int(count_pos)})")
                    break

            if sync_pos[y] == 0:
                if sync_pos[y-1] > 0:
                    sync_pos[y] = sync_pos[y-1]
                updateText(process_label, f"Resyncing image ({y}/{int(height)}, f {int(sync_pos[y])}, {int(count_pos)})")

        for y in range(height):
            image[y] = np.roll(image[y], int(-sync_pos[y]+84), axis=0)
    return image

def fourierFilter(image):
    global filtering_done, image_fourier_pre
    if fourier_filter_checkbox.isChecked():
        for c in range(image.shape[2]):
            image_fourier_pre = np.fft.fftshift(np.fft.fft2(image[:, :, c]))
            for i in range(image_fourier_pre.shape[0]):
                for j in range(image_fourier_pre.shape[1]):
                    # if 1525 < j < 1575 and np.abs(np.real(image_fourier_pre[i, j])) > 10 ** 5:
                    #     image_fourier_pre[i, j] = complex(10 ** 4, 10 ** 4)
                    # if 3925 < j < 3975 and np.abs(np.real(image_fourier_pre[i, j])) > 10 ** 5:
                    #     image_fourier_pre[i, j] = complex(10 ** 4, 10 ** 4)
                    # if 5100 < j < 5200 and np.abs(np.real(image_fourier_pre[i, j])) > 10 ** 5:
                    #     image_fourier_pre[i, j] = complex(10 ** 4, 10 ** 4)
                    # if 300 < j < 400 and np.abs(np.real(image_fourier_pre[i, j])) > 10 ** 5:
                    #     image_fourier_pre[i, j] = complex(10 ** 4, 10 ** 4)
                    if 1525 < j < 1575:
                        image_fourier_pre[i, j] = complex(10 ** 1, 10 ** 1)
                    if 3925 < j < 3975:
                        image_fourier_pre[i, j] = complex(10 ** 1, 10 ** 1)
                    if 5100 < j < 5200:
                        image_fourier_pre[i, j] = complex(10 ** 1, 10 ** 1)
                    if 300 < j < 400:
                        image_fourier_pre[i, j] = complex(10 ** 1, 10 ** 1)
            image[:, :, c] = np.real(np.fft.ifft2(np.fft.ifftshift(image_fourier_pre)))
        filtering_done = True
    return image

def signalToNoise(amplitude):
    SD = amplitude.std(axis=0, ddof=0)
    SNR = np.where(SD == 0, 0, average_amplitude / SD)
    SNR = 20 * np.log10(abs(SNR))
    return SNR

def displayInfo():
    updateText(info_label, f"Image info: Size: {width}x{height},  Length: {round(data.shape[0] / sample_rate, 2)}s,  sample_rate: {sample_rate}Hz,  avAmp: {round(average_amplitude, 2)},  SNR: {round(signalToNoise(amplitude), 2)}dB")
    return

def displayImage(image):
    h, w, _ = image.shape
    img = QImage(image.data, w, h, 3 * w, QImage.Format_RGB888)
    map = QPixmap(img).scaled(image_label.size(), transformMode=Qt.TransformationMode.SmoothTransformation)
    image_label.setPixmap(map)
    return

def save():
    if processing_done:
        input_file_name = input_file.split("/")[-1].split(".")[-2]
        path, check = QFileDialog.getSaveFileName(None, "Save Image", "C:/Users/Jonas/projects/personal/matura/py/out/"+input_file_name+".png", "PNGfile (*.png)")
        if check:
            if aspect_ratio_checkbox.isChecked():
                aspect_ratio_image = transform.resize(image, (int(width/aspect_ratio), width))
                plt.imsave(path, aspect_ratio_image)
            else:
                plt.imsave(path, image)
    return

# --- UI updating ---

def updateText(element, str):
    element.setText(str)
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
    global resample_factor
    if value == "" or value == "-":
        value = 1
    resample_factor = int(value)
    return

def setResampleFactorSlider(value):
    global resample_factor
    resample_factor = 2 ** int(value)
    resample_factor_entry.setText(str(2 ** int(value)))
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
    brightness_entry.setText(str(int(value)/10))
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
    contrast_entry.setText(str(int(value)/10))
    return

def setOffset(value):
    global offset
    if value == "" or value == "." or value == "-":
        value = 0
    offset = float(value)
    return

def setOffsetSlider(value):
    global offset
    offset = float(int(value)/1000)
    offset_entry.setText(str(int(value)/10))
    return

def setAspectRatio(value):
    global aspect_ratio
    if value == "" or value == "." or value == "-":
        value = 1
    aspect_ratio = float(value)
    return

# --- Plots ---

def plotWav():
    if processing_done:
        plt.ion()
        plt.figure(0, figsize=(24, 8))
        plt.plot(data)
        plt.plot(amplitude)
        plt.axhline(y=average_amplitude, color='r', linestyle='-')
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.title(f"{input_file}, {str(original_sample_rate)}Hz, ({str(sample_rate)}), {str(start)}-{str(end)}s")
        plt.show()
    return

def plotWavFourier():
    if processing_done:
        plt.ion()
        plt.figure(1, figsize=(8, 8))
        plt.plot(np.real(np.fft.fftshift(np.fft.fft(data))))
        plt.title("Wav fourier")
        plt.show()
    return

def plotSpectrogram():
    if processing_done:
        plt.ion()
        plt.figure(2, figsize=(8, 8))
        plt.specgram(data)
        plt.title("Spectrogram")
        plt.show()
    return
    
def plotImage():
    if processing_done:
        plt.ion()
        plt.figure(3, figsize=(24, 16))
        plt.imshow(image, aspect=image.shape[1] / image.shape[0] * 0.8)
        plt.title(f"{input_file}, {str(original_sample_rate)}Hz, ({str(sample_rate)}), {str(start)}-{str(end)}s, {str(image.shape[1])}x{str(image.shape[0])}, {str(brightness)}, {str(contrast)}")
        plt.show()
    return

def plotImageFourierPre():
    if processing_done:
        if filtering_done:
            plt.ion()
            plt.figure(4, figsize=(8, 8))
            plt.title("Image fourier real")
            for c in range(image.shape[2]):
                plt.imshow(np.log10(np.abs(np.real(image_fourier_pre))), cmap="gray", aspect=image.shape[1] / image.shape[0] * 0.8)
            plt.figure(5, figsize=(8, 8))
            plt.title("Image fourier imaginary")
            for c in range(image.shape[2]):
                plt.imshow(np.log10(np.abs(np.imag(image_fourier_pre))), cmap="gray", aspect=image.shape[1] / image.shape[0] * 0.8)
            plt.show()
    return

def plotImageFourierPost():
    if processing_done:
        plt.ion()
        plt.figure(6, figsize=(8, 8))
        for c in range(image.shape[2]):
            image_fourier_post = np.fft.fftshift(np.fft.fft2(image[:, :, c]))
            plt.imshow(np.log(abs(image_fourier_post)), cmap="gray", aspect=image.shape[1] / image.shape[0] * 0.8)
        plt.title("Image fourier")
        plt.show()
    return

# --- Event listeners ---

select_file_button.clicked.connect(selectFile)
process_button.clicked.connect(decode)
save_button.clicked.connect(save)
plot_wav_button.clicked.connect(plotWav)
plot_wav_fourier_button.clicked.connect(plotWavFourier)
plot_spectrogram_button.clicked.connect(plotSpectrogram)
plot_image_button.clicked.connect(plotImage)
plot_image_fourier_pre_button.clicked.connect(plotImageFourierPre)
plot_image_fourier_post_button.clicked.connect(plotImageFourierPost)

start_entry.textChanged.connect(setStart)
end_entry.textChanged.connect(setEnd)
shift_entry.textChanged.connect(setShift)
resample_factor_entry.textChanged.connect(setResampleFactor)
brightness_entry.textChanged.connect(setBrightness)
contrast_entry.textChanged.connect(setContrast)
offset_entry.textChanged.connect(setOffset)
aspect_ratio_entry.textChanged.connect(setAspectRatio)

resample_factor_slider.valueChanged.connect(setResampleFactor)
brightness_slider.valueChanged.connect(setBrightnessSlider)
contrast_slider.valueChanged.connect(setContrastSlider)
offset_slider.valueChanged.connect(setOffsetSlider)

app.exec_()
