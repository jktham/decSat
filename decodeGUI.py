import math
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import requests
import scipy.io.wavfile as wav
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from scipy import signal
from skimage import transform

# --- UI setup ---

app = QApplication([])

main_window = QWidget()
main_window.setFixedSize(1800, 1100)
main_window.setWindowTitle("decodeGUI")
main_window.show()

select_file_button = QPushButton("Select file", main_window)
select_file_button.move(10, 10)
select_file_button.resize(200, 40)
select_file_button.show()

select_file_label = QLabel("", main_window)
select_file_label.move(220, 10)
select_file_label.resize(1300, 40)
select_file_label.show()

start_label = QLabel("Start", main_window)
start_label.move(10, 60)
start_label.resize(100, 40)
start_label.show()

start_entry = QLineEdit("0", main_window)
start_entry.setValidator(QDoubleValidator())
start_entry.setAlignment(Qt.AlignRight)
start_entry.move(130, 60)
start_entry.resize(80, 40)
start_entry.show()

end_label = QLabel("End", main_window)
end_label.move(10, 110)
end_label.resize(100, 40)
end_label.show()

end_entry = QLineEdit("0", main_window)
end_entry.setValidator(QDoubleValidator())
end_entry.setAlignment(Qt.AlignRight)
end_entry.move(130, 110)
end_entry.resize(80, 40)
end_entry.show()

shift_label = QLabel("Shift", main_window)
shift_label.move(10, 160)
shift_label.resize(100, 40)
shift_label.show()

shift_entry = QLineEdit("0", main_window)
shift_entry.setValidator(QIntValidator())
shift_entry.setAlignment(Qt.AlignRight)
shift_entry.move(130, 160)
shift_entry.resize(80, 40)
shift_entry.show()

resample_factor_label = QLabel("Resample", main_window)
resample_factor_label.move(10, 210)
resample_factor_label.resize(120, 40)
resample_factor_label.show()

resample_factor_entry = QLineEdit("1", main_window)
resample_factor_entry.setValidator(QIntValidator())
resample_factor_entry.setAlignment(Qt.AlignRight)
resample_factor_entry.move(130, 210)
resample_factor_entry.resize(80, 40)
resample_factor_entry.show()

resample_factor_slider = QSlider(main_window)
resample_factor_slider.setOrientation(Qt.Orientation.Horizontal)
resample_factor_slider.setTickInterval(1)
resample_factor_slider.setTickPosition(3)
resample_factor_slider.setValue(0)
resample_factor_slider.setMinimum(0)
resample_factor_slider.setMaximum(4)
resample_factor_slider.move(10, 260)
resample_factor_slider.resize(200, 40)
resample_factor_slider.show()

brightness_label = QLabel("Brightness", main_window)
brightness_label.move(10, 310)
brightness_label.resize(100, 40)
brightness_label.show()

brightness_entry = QLineEdit("1.0", main_window)
brightness_entry.setValidator(QDoubleValidator())
brightness_entry.setAlignment(Qt.AlignRight)
brightness_entry.move(130, 310)
brightness_entry.resize(80, 40)
brightness_entry.show()

brightness_slider = QSlider(main_window)
brightness_slider.setOrientation(Qt.Orientation.Horizontal)
brightness_slider.setTickInterval(1)
brightness_slider.setTickPosition(3)
brightness_slider.setValue(10)
brightness_slider.setMinimum(5)
brightness_slider.setMaximum(15)
brightness_slider.move(10, 360)
brightness_slider.resize(200, 40)
brightness_slider.show()

contrast_label = QLabel("Contrast", main_window)
contrast_label.move(10, 410)
contrast_label.resize(100, 40)
contrast_label.show()

contrast_entry = QLineEdit("1.0", main_window)
contrast_entry.setValidator(QDoubleValidator())
contrast_entry.setAlignment(Qt.AlignRight)
contrast_entry.move(130, 410)
contrast_entry.resize(80, 40)
contrast_entry.show()

contrast_slider = QSlider(main_window)
contrast_slider.setOrientation(Qt.Orientation.Horizontal)
contrast_slider.setTickInterval(1)
contrast_slider.setTickPosition(3)
contrast_slider.setValue(10)
contrast_slider.setMinimum(5)
contrast_slider.setMaximum(15)
contrast_slider.move(10, 460)
contrast_slider.resize(200, 40)
contrast_slider.show()

offset_label = QLabel("Offset", main_window)
offset_label.move(10, 510)
offset_label.resize(100, 40)
offset_label.show()

offset_entry = QLineEdit("0.0", main_window)
offset_entry.setValidator(QDoubleValidator())
offset_entry.setAlignment(Qt.AlignRight)
offset_entry.move(130, 510)
offset_entry.resize(80, 40)
offset_entry.show()

offset_slider = QSlider(main_window)
offset_slider.setOrientation(Qt.Orientation.Horizontal)
offset_slider.setTickInterval(1)
offset_slider.setTickPosition(3)
offset_slider.setValue(0)
offset_slider.setMinimum(-10)
offset_slider.setMaximum(10)
offset_slider.move(10, 560)
offset_slider.resize(200, 40)
offset_slider.show()

high_pass_filter_label = QLabel("High pass filter", main_window)
high_pass_filter_label.move(10, 610)
high_pass_filter_label.resize(160, 40)
high_pass_filter_label.show()

high_pass_filter_checkbox = QCheckBox(main_window)
high_pass_filter_checkbox.setChecked(True)
high_pass_filter_checkbox.move(180, 610)
high_pass_filter_checkbox.resize(40, 40)
high_pass_filter_checkbox.show()

fourier_filter_label = QLabel("Fourier filter", main_window)
fourier_filter_label.move(10, 660)
fourier_filter_label.resize(160, 40)
fourier_filter_label.show()

fourier_filter_checkbox = QCheckBox(main_window)
fourier_filter_checkbox.setChecked(False)
fourier_filter_checkbox.move(180, 660)
fourier_filter_checkbox.resize(40, 40)
fourier_filter_checkbox.show()

resync_label = QLabel("Resync", main_window)
resync_label.move(10, 710)
resync_label.resize(160, 40)
resync_label.show()

resync_checkbox = QCheckBox(main_window)
resync_checkbox.setChecked(False)
resync_checkbox.move(180, 710)
resync_checkbox.resize(40, 40)
resync_checkbox.show()

get_telemetry_label = QLabel("Get telemetry", main_window)
get_telemetry_label.move(10, 760)
get_telemetry_label.resize(160, 40)
get_telemetry_label.show()

get_telemetry_checkbox = QCheckBox(main_window)
get_telemetry_checkbox.setChecked(False)
get_telemetry_checkbox.move(180, 760)
get_telemetry_checkbox.resize(40, 40)
get_telemetry_checkbox.show()

process_button = QPushButton("Process file", main_window)
process_button.move(10, 1050)
process_button.resize(200, 40)
process_button.show()
process_button.setEnabled(False)

process_label = QLabel("", main_window)
process_label.move(220, 1050)
process_label.resize(600, 40)
process_label.show()

image_label = QLabel("", main_window)
image_label.move(220, 60)
image_label.resize(1360, 980)
image_label.show()

info_label = QLabel("", main_window)
info_label.move(400, 1050)
info_label.resize(1000, 40)
info_label.show()


sat_button = QPushButton("Sat passes", main_window)
sat_button.move(1590, 10)
sat_button.resize(200, 40)
sat_button.show()

plot_wav_button = QPushButton("Plot wav", main_window)
plot_wav_button.move(1590, 60)
plot_wav_button.resize(200, 40)
plot_wav_button.show()
plot_wav_button.setEnabled(False)

plot_wav_envelope_button = QPushButton("Plot wav envelope", main_window)
plot_wav_envelope_button.move(1590, 110)
plot_wav_envelope_button.resize(200, 40)
plot_wav_envelope_button.show()
plot_wav_envelope_button.setEnabled(False)

plot_wav_fourier_button = QPushButton("Plot wav fourier", main_window)
plot_wav_fourier_button.move(1590, 160)
plot_wav_fourier_button.resize(200, 40)
plot_wav_fourier_button.show()
plot_wav_fourier_button.setEnabled(False)

plot_spectrogram_button = QPushButton("Plot spectrogram", main_window)
plot_spectrogram_button.move(1590, 210)
plot_spectrogram_button.resize(200, 40)
plot_spectrogram_button.show()
plot_spectrogram_button.setEnabled(False)

plot_image_button = QPushButton("Plot image", main_window)
plot_image_button.move(1590, 260)
plot_image_button.resize(200, 40)
plot_image_button.show()
plot_image_button.setEnabled(False)

plot_image_fourier_pre_button = QPushButton("Plot fourier (pre)", main_window)
plot_image_fourier_pre_button.move(1590, 310)
plot_image_fourier_pre_button.resize(200, 40)
plot_image_fourier_pre_button.show()
plot_image_fourier_pre_button.setEnabled(False)

plot_image_fourier_post_button = QPushButton("Plot fourier (post)", main_window)
plot_image_fourier_post_button.move(1590, 360)
plot_image_fourier_post_button.resize(200, 40)
plot_image_fourier_post_button.show()
plot_image_fourier_post_button.setEnabled(False)

plot_thermal_image_button = QPushButton("Plot thermal image", main_window)
plot_thermal_image_button.move(1590, 410)
plot_thermal_image_button.resize(200, 40)
plot_thermal_image_button.show()
plot_thermal_image_button.setEnabled(False)

telemetry_label = QLabel("", main_window)
telemetry_label.move(1590, 460)
telemetry_label.resize(200, 480)
telemetry_label.show()

height_correction_checkbox = QCheckBox(main_window)
height_correction_checkbox.setChecked(True)
height_correction_checkbox.move(1590, 950)
height_correction_checkbox.resize(40, 40)
height_correction_checkbox.show()

height_correction_label = QLabel("Height correction", main_window)
height_correction_label.move(1630, 950)
height_correction_label.resize(200, 40)
height_correction_label.show()

aspect_ratio_checkbox = QCheckBox(main_window)
aspect_ratio_checkbox.setChecked(False)
aspect_ratio_checkbox.move(1590, 1000)
aspect_ratio_checkbox.resize(40, 40)
aspect_ratio_checkbox.show()

aspect_ratio_entry_label = QLabel("Ratio", main_window)
aspect_ratio_entry_label.move(1630, 1000)
aspect_ratio_entry_label.resize(200, 40)
aspect_ratio_entry_label.show()

aspect_ratio_entry = QLineEdit("1.4", main_window)
aspect_ratio_entry.setValidator(QDoubleValidator())
aspect_ratio_entry.setAlignment(Qt.AlignRight)
aspect_ratio_entry.move(1710, 1000)
aspect_ratio_entry.resize(80, 40)
aspect_ratio_entry.show()

save_button = QPushButton("Save image", main_window)
save_button.move(1590, 1050)
save_button.resize(200, 40)
save_button.show()
save_button.setEnabled(False)


sat_window = QWidget()
sat_window.setFixedSize(1400, 900)
sat_window.setWindowTitle("Satellite passes")

sat_refresh_button = QPushButton("Refresh", sat_window)
sat_refresh_button.move(10, 10)
sat_refresh_button.resize(200, 40)
sat_refresh_button.show()

sat_refresh_label = QLabel("", sat_window)
sat_refresh_label.move(220, 10)
sat_refresh_label.resize(800, 40)
sat_refresh_label.show()

sat_lat_label = QLabel("Latitude", sat_window)
sat_lat_label.move(10, 60)
sat_lat_label.resize(100, 40)
sat_lat_label.show()

sat_lat_entry = QLineEdit("47.38", sat_window)
sat_lat_entry.setValidator(QDoubleValidator())
sat_lat_entry.setAlignment(Qt.AlignRight)
sat_lat_entry.move(130, 60)
sat_lat_entry.resize(80, 40)
sat_lat_entry.show()

sat_lng_label = QLabel("Longitude", sat_window)
sat_lng_label.move(10, 110)
sat_lng_label.resize(100, 40)
sat_lng_label.show()

sat_lng_entry = QLineEdit("8.54", sat_window)
sat_lng_entry.setValidator(QDoubleValidator())
sat_lng_entry.setAlignment(Qt.AlignRight)
sat_lng_entry.move(130, 110)
sat_lng_entry.resize(80, 40)
sat_lng_entry.show()

sat_tz_label = QLabel("Timezone", sat_window)
sat_tz_label.move(10, 160)
sat_tz_label.resize(100, 40)
sat_tz_label.show()

sat_tz_entry = QLineEdit("1.0", sat_window)
sat_tz_entry.setValidator(QDoubleValidator())
sat_tz_entry.setAlignment(Qt.AlignRight)
sat_tz_entry.move(130, 160)
sat_tz_entry.resize(80, 40)
sat_tz_entry.show()

sat_days_label = QLabel("Days", sat_window)
sat_days_label.move(10, 210)
sat_days_label.resize(100, 40)
sat_days_label.show()

sat_days_entry = QLineEdit("10", sat_window)
sat_days_entry.setValidator(QIntValidator())
sat_days_entry.setAlignment(Qt.AlignRight)
sat_days_entry.move(130, 210)
sat_days_entry.resize(80, 40)
sat_days_entry.show()

sat_days_slider = QSlider(sat_window)
sat_days_slider.setOrientation(Qt.Orientation.Horizontal)
sat_days_slider.setTickInterval(1)
sat_days_slider.setTickPosition(3)
sat_days_slider.setValue(10)
sat_days_slider.setMinimum(1)
sat_days_slider.setMaximum(10)
sat_days_slider.move(10, 260)
sat_days_slider.resize(200, 40)
sat_days_slider.show()

sat_mel_label = QLabel("Min elev.", sat_window)
sat_mel_label.move(10, 310)
sat_mel_label.resize(100, 40)
sat_mel_label.show()

sat_mel_entry = QLineEdit("20", sat_window)
sat_mel_entry.setValidator(QIntValidator())
sat_mel_entry.setAlignment(Qt.AlignRight)
sat_mel_entry.move(130, 310)
sat_mel_entry.resize(80, 40)
sat_mel_entry.show()

sat_mel_slider = QSlider(sat_window)
sat_mel_slider.setOrientation(Qt.Orientation.Horizontal)
sat_mel_slider.setTickInterval(10)
sat_mel_slider.setTickPosition(3)
sat_mel_slider.setValue(20)
sat_mel_slider.setMinimum(0)
sat_mel_slider.setMaximum(90)
sat_mel_slider.move(10, 360)
sat_mel_slider.resize(200, 40)
sat_mel_slider.show()

sat_highlight_label = QLabel("Highlight elev.", sat_window)
sat_highlight_label.move(10, 410)
sat_highlight_label.resize(180, 40)
sat_highlight_label.show()

sat_highlight_checkbox = QCheckBox(sat_window)
sat_highlight_checkbox.setChecked(True)
sat_highlight_checkbox.move(180, 410)
sat_highlight_checkbox.resize(40, 40)
sat_highlight_checkbox.show()

sat_label = QLabel("", sat_window)
sat_label.setAlignment(Qt.AlignTop)
sat_label.setWordWrap(True)
sat_label.setFont(QFont('Courier'))
sat_label.setTextFormat(Qt.RichText)
sat_label.setContentsMargins(8, 8, 8, 8)
sat_label.move(220, 60)
sat_label.resize(1170, 830)
sat_label.show()

sat_label_scroll = QScrollArea(sat_window)
sat_label_scroll.setWidgetResizable(True)
sat_label_scroll.setWidget(sat_label)
sat_label_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
sat_label_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
sat_label_scroll.move(220, 60)
sat_label_scroll.resize(1170, 830)

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
height_multiplier = 2.65

ch_r = 1
ch_g = 1
ch_b = 1

sat_key = "87BT5R-78CLSF-NXGHQ8-4MOH"
sat_id = [25338, 28654, 33591, 40069]
sat_lat = 47.38
sat_lng = 8.54
sat_alt = 0
sat_days = 10
sat_mel = 20
sat_tz = 1.0
sat_transactions = 0

processing_done = False
fourier_done = False

def decode():
    global data, original_sample_rate, sample_rate, amplitude, average_amplitude, image, processing_done, fourier_done
    processing_done = False
    fourier_done = False
    time_start = time.time()    

    updateText(info_label, "")
    updateText(telemetry_label, "")
    process_button.setEnabled(False)
    updateText(process_button, "Processing")
    plot_wav_button.setEnabled(False)
    plot_wav_envelope_button.setEnabled(False)
    plot_wav_fourier_button.setEnabled(False)
    plot_spectrogram_button.setEnabled(False)
    plot_image_button.setEnabled(False)
    plot_image_fourier_pre_button.setEnabled(False)
    plot_image_fourier_post_button.setEnabled(False)
    plot_thermal_image_button.setEnabled(False)
    save_button.setEnabled(False)

    updateText(process_label, "Loading file")
    original_sample_rate, data = wav.read(input_file)

    updateText(process_label, "Resampling data")
    data, sample_rate = resample(data, original_sample_rate, resample_factor, resample_rate)

    updateText(process_label, "Cropping data")
    data = crop(data, start, end, sample_rate)

    if high_pass_filter_checkbox.isChecked():
        updateText(process_label, "Filtering data (High pass)")
        data = highPassFilter(data)

    updateText(process_label, "Generating amplitude envelope")
    amplitude = getEnvelope(data)

    updateText(process_label, "Calculating average amplitude")
    average_amplitude = getAverageAmplitude(amplitude)

    updateText(process_label, "Generating image")
    image = generateImage(amplitude, sample_rate, average_amplitude)

    updateText(process_label, "Offsetting image")
    image = applyOffset(image, offset)

    if resync_checkbox.isChecked():
        updateText(process_label, "Resyncing image")
        image = resync(image)

    if fourier_filter_checkbox.isChecked():
        updateText(process_label, "Filtering image (Fourier)")
        image = fourierFilter(image)
        fourier_done = True

    if get_telemetry_checkbox.isChecked():
        updateText(process_label, "Getting telemetry")
        telemetry_wedges = getTelemetry(image)
        telemetry_string = "\n".join([f"{str(i)}: {str(round(telemetry_wedges[0][i], 2))}, {str(round(telemetry_wedges[1][i], 2))}" for i in range(16)])
        updateText(telemetry_label, "Telemetry (A, B):\n" + telemetry_string)

    updateText(process_label, f"Done ({str(round(time.time() - time_start, 2))}s)")
    processing_done = True

    displayImage(image)
    displayInfo()
    
    process_button.setEnabled(True)
    updateText(process_button, "Process file")
    plot_wav_button.setEnabled(True)
    plot_wav_envelope_button.setEnabled(True)
    plot_wav_fourier_button.setEnabled(True)
    plot_spectrogram_button.setEnabled(True)
    plot_image_button.setEnabled(True)
    plot_image_fourier_pre_button.setEnabled(True)
    plot_image_fourier_post_button.setEnabled(True)
    plot_thermal_image_button.setEnabled(True)
    save_button.setEnabled(True)
    return

def selectFile():
    global input_file
    input_file, check = QFileDialog.getOpenFileName(None, "Select File", "C:/Users/Jonas/projects/personal/matura/py/in", "WAV files (*.wav)")
    if check:
        sr, _ = wav.read(input_file)
        updateText(select_file_label, f"{input_file}, {sr}Hz")
        process_button.setEnabled(True)
    else:
        updateText(select_file_label, "")
        process_button.setEnabled(False)
    return

def resample(data, sample_rate, resample_factor, resample_rate):
    # data = signal.resample(data, int(data.shape[0] / sample_rate) * 20800)
    # data = signal.decimate(data, 5)
    # sample_rate = resample_rate

    # coef = 20800 / sample_rate
    # samples = int(coef * len(data))
    # data = signal.resample(data, samples)
    # sample_rate = 20800

    data = data[::resample_factor]
    sample_rate = int(sample_rate / resample_factor)
    return data, sample_rate

def crop(data, start, end, sample_rate):
    if end > start and (start > 0 or end > 0):
        startSample = int(start * sample_rate)
        endSample = int(end * sample_rate)
        data = data[startSample:endSample]
    return data

# TODO: Fix for low samplerate
def highPassFilter(data):
    passes = 1
    hpf = signal.firwin(101, min(1200, sample_rate/2 - 1), fs=sample_rate, pass_zero=False)
    for i in range(passes):
        data = signal.lfilter(hpf, [1.0], data)
    return data

def getEnvelope(data):
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
        image[y, x] = (int(lum*ch_r), int(lum*ch_g), int(lum*ch_b))
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
    side_a = np.zeros(height)
    side_b = np.zeros(height)
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

            k = int(width/45)
            dark_sum = np.sum(np.take(image[y], range(x, x+k), axis=0, mode="wrap"), axis=0)[0]
            bright_sum = np.sum(np.take(image[y], range(int(x+width/2), int(x+k+width/2)), axis=0, mode="wrap"), axis=0)[0]

            if (dark_sum < k * 40 and bright_sum > k * 240): # or (dark_sum > k * 220 and bright_sum > k * 220) or (dark_sum < k * 50 and bright_sum < k * 50):
                sync_pos[y] = x
                count_pos += 1
                updateText(process_label, f"Resyncing image ({y}/{int(height)}, {int(sync_pos[y])}, {int(count_pos)}, success)")
                break

        if sync_pos[y] == 0:
            if sync_pos[y-1] > 0:
                sync_pos[y] = sync_pos[y-1]
            updateText(process_label, f"Resyncing image ({y}/{int(height)}, {int(sync_pos[y])}, {int(count_pos)}, fallback)")

    for y in range(height):
        image[y] = np.roll(image[y], int(-sync_pos[y]+width/65), axis=0)
        side_a[y] = np.sum(image[y, :int(width/2), 0])
        side_b[y] = np.sum(image[y, int(width/2+1):, 0])
        if side_a[y] > side_b[y]:
            image[y] = np.roll(image[y], int(width/2), axis=0)
    return image

def getTelemetry(image):
    telemetry = np.zeros((2, height))
    telemetry_wedges = np.zeros((2, 16))
    telemetry_edges = np.zeros((2, 8))
    edge_index = np.zeros(2)
    b = width/551

    for y in range(10, height-10):
        telemetry[0][y] = np.sum(image[y, int(width/2.094+b):int(width/2.003-b), 0]) / (int(width/2.003-b) - int(width/2.094+b))
        telemetry[1][y] = np.sum(image[y, int(width/1.023+b):int(width/1.001-b), 0]) / (int(width/1.001-b) - int(width/1.023+b))
    
    for c in range(2):

        for i in range(8):
            for j in range(len(telemetry[c]) // 8):
                telemetry_edges[c][i] += abs(telemetry[c][i-1+8*j] - telemetry[c][i+8*j])
        
        edge_index[c] = np.argmax(abs(telemetry_edges[c]))

        for i in range(16):
            for j in range(len(telemetry[c]) // 128):
                telemetry_wedges[c][i] += np.sum(telemetry[c][int(edge_index[c]) + i*8 + j*128 : int(edge_index[c]) + i*8+8 + j*128])
            telemetry_wedges[c][i] /= 8 * (len(telemetry[c]) // 128)

        telemetry_wedges[c] = np.roll(telemetry_wedges[c], np.argmax(telemetry_wedges[c]) + 15)
    
    return telemetry_wedges

def fourierFilter(image):
    global image_fourier_pre, fourier_done
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
    fourier_done = True
    return image

def signalToNoise(amplitude):
    sd = amplitude.std(axis=0, ddof=0)
    snr = np.where(sd == 0, 0, average_amplitude / sd)
    snr = 20 * np.log10(abs(snr))
    return snr

def displayInfo():
    updateText(info_label, f"Size: {width}x{height},  Length: {round(data.shape[0] / sample_rate, 2)}s,  SR: {sample_rate}Hz,  avAmp: {round(average_amplitude, 2)},  SNR: {round(signalToNoise(amplitude), 2)}dB")
    return

def displayImage(image):
    h, w, _ = image.shape
    img = QImage(image.data, w, h, 3 * w, QImage.Format_RGB888)
    map = QPixmap(img).scaled(image_label.size(), transformMode=Qt.TransformationMode.SmoothTransformation)
    image_label.setPixmap(map)
    return

def saveImage():
    if processing_done:
        input_file_name = input_file.split("/")[-1].split(".")[-2]
        path, check = QFileDialog.getSaveFileName(None, "Save Image", f"C:/Users/Jonas/projects/personal/matura/py/out/{input_file_name}.png", "PNGfile (*.png)")
        if check:
            if aspect_ratio_checkbox.isChecked():
                resized_img = transform.resize(image, (int(width/aspect_ratio), width))
                plt.imsave(path, resized_img)
            elif height_correction_checkbox.isChecked():
                resized_img = transform.resize(image, (int(height*(width/2080)), width))
                plt.imsave(path, resized_img)
            else:
                plt.imsave(path, image)
                
    return

def showSat():
    sat_window.show()
    app.processEvents()
    if sat_transactions == 0:
        refreshSat()
    return

def refreshSat():
    global sat_transactions
    sat_response = [None] * len(sat_id)

    sat_refresh_button.setEnabled(False)
    sat_window.setFocus()
    updateText(sat_refresh_button, "Refreshing")

    for i in range(len(sat_id)):
        try:
            sat_response[i] = requests.get(f"https://api.n2yo.com/rest/v1/satellite/radiopasses/{str(sat_id[i])}/{str(sat_lat)}/{str(sat_lng)}/{str(sat_alt)}/{str(sat_days)}/{str(sat_mel)}/&apiKey={sat_key}")
        except requests.exceptions.RequestException as e:
            updateText(sat_label, str(e))
            sat_refresh_button.setEnabled(True)
            sat_refresh_button.setFocus()
            updateText(sat_refresh_button, "Refresh")
            return
        sat_response[i] = sat_response[i].json()

    sat_length = [0] * len(sat_response)

    for i in range(len(sat_response)):
        for j in range(len(sat_response) - i):
            sat_length[i] += sat_response[j]["info"]["passescount"]
    sat_length = sat_length[::-1]

    sat_passes = [None] * sat_length[-1]
    for i in range(sat_length[-1]):
        for j in range(len(sat_length)):
            if j == 0:
                c = 0
            else:
                c = sat_length[j-1]
            if c <= i < sat_length[j]:
                sat_passes[i] = sat_response[j]["passes"][i - c]
                sat_passes[i]["index"] = i
                sat_passes[i]["satname"] = sat_response[j]["info"]["satname"]

    for i in range(len(sat_passes)):
        sat_passes[i]["startUTC"] += int(3600 * sat_tz)
        sat_passes[i]["maxUTC"] += int(3600 * sat_tz)
        sat_passes[i]["endUTC"] += int(3600 * sat_tz)

    sat_passes = sorted(sat_passes, key=lambda k: k["startUTC"])
    
    sat_names = [None] * len(sat_response)
    for i in range(len(sat_response)):
        sat_names[i] = sat_response[i]["info"]["satname"]
    
    sat_string = ""
    for i in range(len(sat_passes)):
        sat_string_date = f"{datetime.utcfromtimestamp(sat_passes[i]['startUTC']).strftime('%Y-%m-%d')}"
        sat_string_time = f"{datetime.utcfromtimestamp(sat_passes[i]['startUTC']).strftime('%H:%M:%S')}_{datetime.utcfromtimestamp(sat_passes[i]['maxUTC']).strftime('%H:%M:%S')}_{datetime.utcfromtimestamp(sat_passes[i]['endUTC']).strftime('%H:%M:%S')}"
        sat_string_mel = f"{'_' * (5 - len(str(format(sat_passes[i]['maxEl'], '.2f'))))}{format(sat_passes[i]['maxEl'], '.2f')}"
        sat_string_melc = f"{sat_passes[i]['maxAzCompass']}{'_' * (3 - len(sat_passes[i]['maxAzCompass']))}"
        sat_string_name = f"{sat_passes[i]['satname']}{'_' * (len(max(sat_names, key=len)) - len(sat_passes[i]['satname']))}"
        sat_string_az = f"{'_' * (6 - len(str(format(sat_passes[i]['startAz'], '.2f'))))}{format(sat_passes[i]['startAz'], '.2f')}__{'_' * (6 - len(str(format(sat_passes[i]['maxAz'], '.2f'))))}{format(sat_passes[i]['maxAz'], '.2f')}__{'_' * (6 - len(str(format(sat_passes[i]['endAz'], '.2f'))))}{format(sat_passes[i]['endAz'], '.2f')}"
        
        if math.cos(sat_passes[i]["startAz"] / 360 * 2*math.pi) < math.cos(sat_passes[i]["endAz"] / 360 * 2*math.pi):
            sat_string_dir = "N"
        else:
            sat_string_dir = "S"

        sat_string_alpha = (sat_passes[i]['maxEl']-45)/45
        if sat_string_alpha < 0:
            sat_string_alpha = 0
        if not sat_highlight_checkbox.isChecked():
            sat_string_alpha = 0
        sat_string_color = (255, 255, 0, sat_string_alpha)

        if datetime.utcfromtimestamp(sat_passes[i]['startUTC']).date() > datetime.utcfromtimestamp(sat_passes[i-1]['startUTC']).date():
            sat_string += "<br>"
        sat_string += f"{sat_string_date}__{sat_string_time}____{sat_string_name}____<span style=\"background-color: rgba{sat_string_color}\">{sat_string_mel}</span>_{sat_string_dir}_{sat_string_melc}____{sat_string_az}<br>"
    sat_string = sat_string.replace("_", "&nbsp;")

    sat_transactions = sat_response[-1]["info"]["transactionscount"]
    sat_passescount = [None] * len(sat_response)
    for i in range(len(sat_response)):
        sat_passescount[i] = sat_response[i]["info"]["passescount"]

    sat_refresh_button.setEnabled(True)
    sat_refresh_button.setFocus()
    updateText(sat_refresh_button, "Refresh")
    updateText(sat_refresh_label, f"{str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}, Transactions: {str(sat_transactions)}, Passes: {str(sat_passescount)}")
    updateText(sat_label, sat_string)
    sat_label.adjustSize()
    return

# --- Plots ---

def plotWav():
    if processing_done:
        plt.ion()
        plt.figure(0, figsize=(24, 8))
        plt.plot(data)
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.title(f"{input_file}, {str(original_sample_rate)}Hz, ({str(sample_rate)}), {str(start)}-{str(end)}s")
        plt.show()
    return

def plotWavEnvelope():
    if processing_done:
        plt.ion()
        plt.figure(0, figsize=(24, 8))
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
        if fourier_done:
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

def plotThermalImage():
    if processing_done:
        plt.ion()
        plt.figure(7, figsize=(24, 16))
        # colormap = plt.get_cmap("inferno")
        # thermal_image = (colormap(image) * 2**16).astype(np.uint16)[:,:,:3]
        # thermal_image = cv2.cvtColor(thermal_image, cv2.COLOR_RGB2BGR)
        # # thermal_image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
        thermal_image = image[:,:,0]
        plt.imshow(thermal_image, aspect=thermal_image.shape[1] / thermal_image.shape[0] * 0.8, cmap="jet")
        plt.clim(100, 255)
        plt.colorbar()
        plt.title("Thermal image")
        plt.show()
    return

# --- UI updating ---

def updateText(element, str):
    element.setText(str)
    app.processEvents()
    return

def setStart(value):
    global start
    if value == "" or value == "." or value == "-":
        value = 0.0
    start = float(value)
    return

def setEnd(value):
    global end
    if value == "" or value == "." or value == "-":
        value = 0.0
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
    updateText(resample_factor_entry, str(int(2 ** int(value))))
    return

def setBrightness(value):
    global brightness
    if value == "" or value == "." or value == "-":
        value = 1.0
    brightness = float(value)
    return

def setBrightnessSlider(value):
    global brightness
    brightness = float(value/1000)
    updateText(brightness_entry, str(int(value)/10))
    return

def setContrast(value):
    global contrast
    if value == "" or value == "." or value == "-":
        value = 1.0
    contrast = float(value)
    return

def setContrastSlider(value):
    global contrast
    contrast = float(int(value)/1000)
    updateText(contrast_entry, str(int(value)/10))
    return

def setOffset(value):
    global offset
    if value == "" or value == "." or value == "-":
        value = 0.0
    offset = float(value)
    return

def setOffsetSlider(value):
    global offset
    offset = float(int(value)/1000)
    updateText(offset_entry, str(int(value)/10))
    return

def setAspectRatio(value):
    global aspect_ratio
    if value == "" or value == "." or value == "-":
        value = 1.0
    aspect_ratio = float(value)
    return

def setSatLat(value):
    global sat_lat
    if value == "" or value == "." or value == "-":
        value = 0.0
    sat_lat = float(value)
    return

def setSatLng(value):
    global sat_lng
    if value == "" or value == "." or value == "-":
        value = 0.0
    sat_lng = float(value)
    return
    
def setSatTz(value):
    global sat_tz
    if value == "" or value == "." or value == "-":
        value = 0.0
    sat_tz = float(value)
    return

def setSatDays(value):
    global sat_days
    if value == "" or value == "." or value == "-":
        value = 1
    if int(value) > 10:
        value = 10
    if int(value) < 1:
        value = 1
    sat_days = int(value)
    return

def setSatDaysSlider(value):
    global sat_days
    sat_days = int(value)
    updateText(sat_days_entry, str(int(value)))
    return

def setSatMel(value):
    global sat_mel
    if value == "" or value == "." or value == "-":
        value = 0
    if int(value) > 90:
        value = 90
    if int(value) < 0:
        value = 0
    sat_mel = int(value)
    return

def setSatMelSlider(value):
    global sat_mel
    sat_mel = int(value)
    updateText(sat_mel_entry, str(int(value)))
    return

# --- Event listeners ---

select_file_button.clicked.connect(selectFile)
process_button.clicked.connect(decode)
save_button.clicked.connect(saveImage)
plot_wav_button.clicked.connect(plotWav)
plot_wav_envelope_button.clicked.connect(plotWavEnvelope)
plot_wav_fourier_button.clicked.connect(plotWavFourier)
plot_spectrogram_button.clicked.connect(plotSpectrogram)
plot_image_button.clicked.connect(plotImage)
plot_image_fourier_pre_button.clicked.connect(plotImageFourierPre)
plot_image_fourier_post_button.clicked.connect(plotImageFourierPost)
plot_thermal_image_button.clicked.connect(plotThermalImage)
sat_button.clicked.connect(showSat)
sat_refresh_button.clicked.connect(refreshSat)

start_entry.textChanged.connect(setStart)
end_entry.textChanged.connect(setEnd)
shift_entry.textChanged.connect(setShift)
resample_factor_entry.textChanged.connect(setResampleFactor)
brightness_entry.textChanged.connect(setBrightness)
contrast_entry.textChanged.connect(setContrast)
offset_entry.textChanged.connect(setOffset)
aspect_ratio_entry.textChanged.connect(setAspectRatio)
sat_lat_entry.textChanged.connect(setSatLat)
sat_lng_entry.textChanged.connect(setSatLng)
sat_tz_entry.textChanged.connect(setSatTz)
sat_days_entry.textChanged.connect(setSatDays)
sat_mel_entry.textChanged.connect(setSatMel)

resample_factor_slider.valueChanged.connect(setResampleFactorSlider)
brightness_slider.valueChanged.connect(setBrightnessSlider)
contrast_slider.valueChanged.connect(setContrastSlider)
offset_slider.valueChanged.connect(setOffsetSlider)
sat_days_slider.valueChanged.connect(setSatDaysSlider)
sat_mel_slider.valueChanged.connect(setSatMelSlider)

app.exec_()
