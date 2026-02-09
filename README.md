# LVS

This repository is a collection of code to experiment and validate a signal processing/communication pipeline for the LVS project in Python. It includes ADC input, TX/RX, CZT (Chirp z-Transform), FIR/IIR filter design, and demo scripts.

## Key Files
- `main.py`: Project entry point (script combining execution flow)
- `lvs_adc.py`: ADC input/preprocessing logic
- `lvs_tx.py`: Transmit (TX) path logic
- `lvs_rx.py`: Receive (RX) path logic
- `lvs_czt.py`: CZT implementation
- `czt_debug.py`, `czt_debug_continous.py`, `czt_debug_stage.py`:  
CZT debug/validation scripts.
MCU must output input files in the same format as used in the code.
- `fir_design.py`: FIR filter design script
- `stage_demo.py`, `stage_demo_2.py`: Stage-by-stage demo execution scripts
- `observer.py`: Observation/logging/monitoring utility logic
- `config.py`: Common configuration

## Data/Result Files
- `fir_tap*.txt`, `iir_tap.txt`: Filter tap coefficients

## Note
- Modes `full` and `subsample` exist, but validation was performed only for `subsample`.  
: The 1024-sample `adc` input is aligned only for `subsample`.  
For `full`, the TX chunk length needs to be adjusted.
- In `config.py`, `idx = 3` is the final result value.


