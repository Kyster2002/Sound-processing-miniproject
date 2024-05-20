import numpy as np
from scipy.io import wavfile
import librosa

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


input_file = "guitar.wav"

input_data, input_rate = librosa.load(input_file, sr=None, mono=True)

# Adjusted parameters for chapel-like reverb
comb_delay_times = [1500, 2000, 2500, 3000]
comb_feedback_gains = [0.9, 0.9, 0.9, 0.9]
allpass_delay_times = [1000, 1500, 2000]
allpass_feedback_gains = [0.9, 0.9, 0.9]
attenuation_factor = 0.8
plain_delay_time = 3000
plain_reverb_decay = 0.9

# Apply input signal to each comb filter
reverb_signal = np.zeros(len(input_data))
for i in range(len(comb_delay_times)):
    comb_output = comb_filter(input_data, comb_delay_times[i], comb_feedback_gains[i])
    reverb_signal += comb_output

# Apply input signal to each allpass filter
for i in range(len(allpass_delay_times)):
    allpass_output = allpass_filter(reverb_signal, allpass_delay_times[i], allpass_feedback_gains[i])
    reverb_signal = allpass_output

# Apply attenuation
reverb_signal = apply_attenuation(reverb_signal, attenuation_factor)

# Apply plain reverb
plain_reverb_output = plain_reverb(input_data, plain_reverb_decay)

# Apply fade-out to the reverb tail
fade_out_duration = 0.5  # Duration of the fade-out in seconds
fade_out_samples = int(fade_out_duration * input_rate)
fade_out_window = np.linspace(1, 0, fade_out_samples)
plain_reverb_output[-fade_out_samples:] *= fade_out_window

# Add plain reverb output to the reverb signal
reverb_signal += plain_reverb_output

# Normalize the reverb signal
reverb_signal /= np.max(np.abs(reverb_signal))

# Output the final reverb signal
output_signal = reverb_signal

# Apply plain reverb
plain_reverb_output = plain_reverb(input_data, plain_reverb_decay)
reverb_signal += plain_reverb_output

# Normalize the output signal
output_signal = reverb_signal / np.max(np.abs(reverb_signal))

# Write output WAV file
wavfile.write("output.wav", input_rate, output_signal.astype(np.float32))
