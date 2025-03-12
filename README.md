# Convolutional Code Simulation (2,1,6)

## Overview
This project is a C++ simulation of a (2,1,6) convolutional encoder and Viterbi decoder. The program simulates data transmission through an Additive White Gaussian Noise (AWGN) channel and evaluates the Bit Error Rate (BER) performance of the decoding process.

## Features
- **Convolutional Encoding**: Implements a (2,1,6) convolutional encoder.
- **Modulation**: Uses BPSK (Binary Phase Shift Keying) modulation.
- **AWGN Channel Simulation**: Adds Gaussian noise to the transmitted signal.
- **Viterbi Decoding**: Performs soft-decision decoding using the Viterbi algorithm.
- **BER Calculation**: Evaluates the bit error rate across different signal-to-noise ratios (SNRs).

## Installation
### Prerequisites
- C++ compiler (GCC, Clang, or MSVC)
- C++ standard library

### Compilation
Use the following command to compile the program:
```sh
 g++ -o convolutional_simulation main.cpp -O2
```

## Usage
Run the compiled executable:
```sh
./convolutional_simulation
```
The program will simulate transmission over various SNR values and display the BER results.

## File Structure
```
.
├── main.cpp           # Main program file
├── encoder.h/.cpp     # Convolutional encoder implementation
├── decoder.h/.cpp     # Viterbi decoder implementation
├── awgn.h/.cpp        # AWGN channel simulation
├── utils.h/.cpp       # Utility functions
└── README.md          # Project documentation
```

## Example Output
```
SNR: 0 dB, BER: 0.245
SNR: 1 dB, BER: 0.198
SNR: 2 dB, BER: 0.152
...
```

## Future Improvements
- Implement different constraint lengths and generator polynomials.
- Optimize Viterbi decoding for better performance.
- Add visualization tools for BER analysis.
