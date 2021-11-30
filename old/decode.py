import math
import os

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft
import scipy.io.wavfile as wav
from scipy import signal
from skimage import color, transform
from skimage.color.colorconv import gray2rgb

# parameters
inputFile = "C:/Users/Jonas/projects/personal/matura/py/test5_wx.wav"
outputFile = "C:/Users/Jonas/projects/personal/matura/py/output.jpg"
outputFile_r = "C:/Users/Jonas/projects/personal/matura/py/output_r.jpg"
start = 0
end = 0
resample = 1
shift = 0
brightness = 0.8
contrast = 1.3
threshold = 10000

# get file and resample
originalSampleRate, data = wav.read(inputFile)
data = data[::resample]
sampleRate = int(originalSampleRate / resample)
print("resampled data")

# crop data
if start > 0 or end > 0:
    startSample = int(start * sampleRate)
    endSample = int(end * sampleRate)
    data = data[startSample:endSample]
print("cropped data")

# filter wav
# fft_data = np.fft.fft(data)
# fft_data_sum = 0
# for i in range(fft_data.shape[0]):
#     fft_data_sum += fft_data[i]
# fft_data_mean = np.real(fft_data_sum / fft_data.shape[0])
# # for i in range(fft_data.shape[0]):
# #     if abs(np.real(fft_data[i])) > threshold * fft_data_mean:
# #         fft_data[i] = fft_data_mean
# for i in range(fft_data.shape[0]):
#     if 10000 < i < 1000000:
#         fft_data[i] = fft_data_mean
# data = np.real(np.fft.ifft(fft_data))
# print("filtered data")

# get amplitude envelope with hilbert transformation
amplitude = np.abs(signal.hilbert(data))

# create image
width = int(0.5 * (sampleRate + shift))
height = int(amplitude.shape[0] / width)
image = np.zeros((height, width, 3), dtype="uint8")

# equalize brightness
amplitude_sum = 0
for i in range(amplitude.shape[0]):
    amplitude_sum += amplitude[i]
averageAmplitude = int(amplitude_sum / amplitude.shape[0])
print("found average amplitude")

# set pixel luminosity
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
            print("drew line " + str(y) + "/" + str(height) + " (" + str(x) + ")")
        x = 0
        y += 1
        if y >= height:
            break

# plot wav
plt.figure(0, figsize=(12, 4))
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
print("generated wav plot (" + str(sampleRate) + ")")

# plot wav fourier
plt.figure(1, figsize=(4, 4))
plt.plot(np.real(np.fft.fftshift(np.fft.fft(data))))
plt.title("Wav fourier")

# plot image
plt.figure(2, figsize=(12, 8))
plt.imshow(image, aspect=width/height*0.8)
plt.title(
    inputFile + ", " + 
    str(originalSampleRate) + "Hz, (" + 
    str(sampleRate) + "), " + 
    str(start) + "-" + 
    str(end) + "s, " + 
    str(width) + "x" + 
    str(height) + ", " + 
    str(brightness) + ", " + 
    str(contrast)
    )
print("generated image (" + str(width) + "x" + str(height) + ")")

# plot image fourier
plt.figure(3, figsize=(4, 4))
image_fourier = np.fft.fftshift(np.fft.fft2(color.rgb2gray(image)))
plt.imshow(np.log(abs(image_fourier)), cmap="gray", aspect=width/height*0.8)
plt.title("Image fourier")

# save images, display plots
if os.path.isfile(outputFile):
    os.remove(outputFile)

if os.path.isfile(outputFile_r):
    os.remove(outputFile_r)

plt.imsave(outputFile, image)
plt.imsave(outputFile_r, transform.resize(image, (2160, 3840)))
plt.show()
