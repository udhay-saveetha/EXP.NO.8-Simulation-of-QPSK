# EXP.NO.8-Simulation-of-QPSK
## AIM
To analyse the modulation of QPSK Signal

## SOFTWARE REQUIRED
Python

## ALGORITHMS
Make Random Bits

→ Create a list of 0s and 1s (2 bits for each symbol).

Group the Bits

→ Take every 2 bits and turn them into 1 symbol.

Assign Angles

→ Each symbol gets a different angle:

00 → 0°

01 → 90°

10 → 180°

11 → 270°

Make the Signal

→ For each symbol, draw a wave with the right angle.

Join the Waves

→ Put all the waves together to make the full QPSK signal.

Draw the Signal

→ Show the signal using plots (real part, imaginary part, full signal).

## PROGRAM
``` python
import numpy as np
import matplotlib.pyplot as plt

Parameters
num_symbols = 10          # Number of QPSK symbols
T = 1.0                   # Symbol period (s)
fs = 100.0                # Sampling frequency (Hz)
t = np.arange(0, T, 1/fs) # Time vector for one symbol

# Generate random bit sequence
bits = np.random.randint(0, 2, num_symbols * 2)  # 2 bits per QPSK symbol
symbols = 2 * bits[0::2] + bits[1::2]            # Map bits to QPSK symbols (00, 01, 10, 11 → 0, 1, 2, 3)

# QPSK phase mapping
symbol_phases = {
    0: 0,
    1: np.pi / 2,
    2: np.pi,
    3: 3 * np.pi / 2
}

# Initialize QPSK signal and symbol time markers
qpsk_signal = np.array([])
symbol_times = []

# Generate QPSK-modulated signal
for i, symbol in enumerate(symbols):
    phase = symbol_phases[symbol]
    symbol_time = i * T
    qpsk_segment = np.cos(2 * np.pi * t / T + phase) + 1j * np.sin(2 * np.pi * t / T + phase)
    qpsk_signal = np.concatenate((qpsk_signal, qpsk_segment))
    symbol_times.append(symbol_time)

# Time vector for entire signal
t_total = np.arange(0, num_symbols * T, 1/fs)

# Plotting
plt.figure(figsize=(14, 12))

# In-phase component
plt.subplot(3, 1, 1)
plt.plot(t_total, np.real(qpsk_signal), label='In-phase')
for i, symbol_time in enumerate(symbol_times):
    plt.axvline(symbol_time, color='red', linestyle='--', linewidth=0.5)
    plt.text(symbol_time + T/4, 0.5, f'{symbols[i]:02b}', fontsize=12, color='blue')
plt.title('QPSK Signal - In-phase Component with Symbols')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# Quadrature component
plt.subplot(3, 1, 2)
plt.plot(t_total, np.imag(qpsk_signal), label='Quadrature', color='orange')
for i, symbol_time in enumerate(symbol_times):
    plt.axvline(symbol_time, color='red', linestyle='--', linewidth=0.5)
    plt.text(symbol_time + T/4, 0.5, f'{symbols[i]:02b}', fontsize=12, color='blue')
plt.title('QPSK Signal - Quadrature Component with Symbols')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# Resultant QPSK waveform (real part only)
plt.subplot(3, 1, 3)
plt.plot(t_total, np.real(qpsk_signal), label='Resultant QPSK Waveform', color='green')
for i, symbol_time in enumerate(symbol_times):
    plt.axvline(symbol_time, color='red', linestyle='--', linewidth=0.5)
    plt.text(symbol_time + T/4, 0.5, f'{symbols[i]:02b}', fontsize=12, color='blue')
plt.title('Resultant QPSK Waveform')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
```
## OUTPUT
![image](https://github.com/user-attachments/assets/baaddec1-65b1-4ff3-99e3-737b06e2bb3f)

## RESULT / CONCLUSIONS
Thus QPSK modulation is implemented using Python code.
