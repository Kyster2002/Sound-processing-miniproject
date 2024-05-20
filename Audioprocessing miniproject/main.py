import numpy as np
import sounddevice as sd
import librosa
import threading
import tkinter as tk
from scipy.io import wavfile

# Function to create a comb filter
def comb_filter(input_signal, delay_time, feedback_gain):
    output = np.zeros(len(input_signal) + delay_time)
    output[:len(input_signal)] = input_signal

    for i in range(delay_time, len(output)):
        output[i] += feedback_gain * output[i - delay_time]

    return output[:len(input_signal)]

# Function to create an allpass filter
def allpass_filter(input_signal, delay_time, feedback_gain):
    output = np.zeros(len(input_signal) + delay_time)
    output[:len(input_signal)] = input_signal

    for i in range(delay_time, len(output)):
        output[i] = -feedback_gain * output[i - delay_time] + input_signal[i - delay_time] + feedback_gain * output[i]

    return output[:len(input_signal)]

# Function to apply attenuation
def apply_attenuation(input_signal, attenuation_factor):
    return input_signal * attenuation_factor

# Function to create a plain delay
def plain_delay(input_signal, delay_time):
    output = np.zeros(len(input_signal))
    output[delay_time:] = input_signal[:-delay_time]
    return output

# Function to create a plain reverberator
def plain_reverb(input_signal, decay):
    output = np.zeros(len(input_signal))
    for i in range(1, len(input_signal)):
        output[i] = input_signal[i] + decay * output[i - 1]
    return output

class ReverbApp:
    def __init__(self, root, input_file):
        self.root = root
        self.input_file = input_file
        self.input_data, self.input_rate = librosa.load(input_file, sr=None, mono=True)
        self.output_signal = self.input_data.copy()

        self.delay_time = tk.IntVar(value=2000)
        self.feedback_gain = tk.DoubleVar(value=0.9)
        self.attenuation_factor = tk.DoubleVar(value=0.8)
        self.reverb_decay = tk.DoubleVar(value=0.9)

        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="Delay Time (ms)").pack()
        tk.Scale(self.root, from_=0, to=5000, orient="horizontal", variable=self.delay_time).pack()

        tk.Label(self.root, text="Feedback Gain").pack()
        tk.Scale(self.root, from_=0.0, to=1.0, orient="horizontal", resolution=0.01, variable=self.feedback_gain).pack()

        tk.Label(self.root, text="Attenuation Factor").pack()
        tk.Scale(self.root, from_=0.0, to=1.0, orient="horizontal", resolution=0.01, variable=self.attenuation_factor).pack()

        tk.Label(self.root, text="Reverb Decay").pack()
        tk.Scale(self.root, from_=0.0, to=1.0, orient="horizontal", resolution=0.01, variable=self.reverb_decay).pack()

        tk.Button(self.root, text="Play", command=self.play_audio).pack()

    def process_audio(self):
        delay_time = self.delay_time.get()
        feedback_gain = self.feedback_gain.get()
        attenuation_factor = self.attenuation_factor.get()
        reverb_decay = self.reverb_decay.get()

        reverb_signal = np.zeros(len(self.input_data))
        comb_output = comb_filter(self.input_data, delay_time, feedback_gain)
        reverb_signal += comb_output
        allpass_output = allpass_filter(reverb_signal, delay_time, feedback_gain)
        reverb_signal = allpass_output
        reverb_signal = apply_attenuation(reverb_signal, attenuation_factor)
        plain_reverb_output = plain_reverb(self.input_data, reverb_decay)
        reverb_signal += plain_reverb_output
        reverb_signal /= np.max(np.abs(reverb_signal))

        self.output_signal = reverb_signal

    def audio_callback(self, outdata, frames, time, status):
        if status:
            print(status)
        outdata[:] = self.output_signal[:frames].reshape(-1, 1)
        self.output_signal = np.roll(self.output_signal, -frames)

    def play_audio(self):
        self.process_audio()
        with sd.OutputStream(channels=1, callback=self.audio_callback, samplerate=self.input_rate):
            sd.sleep(int(len(self.input_data) / self.input_rate * 1000))

if __name__ == "__main__":
    root = tk.Tk()
    app = ReverbApp(root, "guitar.wav")
    root.mainloop()
