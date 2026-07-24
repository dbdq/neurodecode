<div align="center">

# Neurodecode

**A real-time brain signal decoding framework with a modular software design.**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/neurodecode.svg)](https://pypi.org/project/neurodecode/)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows-lightgrey.svg)]()
[![Built on LSL](https://img.shields.io/badge/built%20on-Lab%20Streaming%20Layer-orange.svg)](https://github.com/sccn/labstreaminglayer)

</div>

---

## Overview

Neurodecode provides a real-time brain signal decoding framework with a modular software design. Its decoding performance was recognised at the [Microsoft Brain Signal Decoding competition](https://github.com/dbdq/microsoft_decoding) with the **First Prize Award** (2016) for high decoding accuracy — 2nd out of 1863 algorithms.

It has been applied to online decoding projects across a wide range of electrode types — **EEG, ECoG, DBS, and microelectrode arrays** — and acquisition systems, including AntNeuro eego, g.tec gUSBamp, BioSemi ActiveTwo, BrainProducts actiCHamp, and Wearable Sensing.

Decoding runs at roughly **15 classifications per second (cps)** on a 4th-gen i7 laptop with a 64-channel setup at 512 Hz. High-speed decoding of up to **200 cps** has been achieved using a process-interleaving technique on 8 cores. Neurodecode is tested on both Linux and Windows with Python 3.8+.

The underlying data communication is built on the [Lab Streaming Layer (LSL)](https://github.com/sccn/labstreaminglayer), which provides sub-millisecond time-synchronization accuracy. Any signal-acquisition system supported natively by LSL or by OpenVibe is supported by Neurodecode. Because the transport is TCP-based, signals can also be transmitted wirelessly.

## Modules

<table>
<thead>
<tr><th align="left" width="1%" nowrap>Module</th><th align="left">Description</th></tr>
</thead>
<tbody>
<tr><td valign="top" nowrap><b>StreamReceiver</b></td><td>Base module for acquiring signals, used by Decoder, StreamViewer, and StreamRecorder.</td></tr>
<tr><td valign="top" nowrap><b>StreamViewer</b></td><td>Visualize signals in real time with spectral filtering, common-average filtering, and real-time FFT.</td></tr>
<tr><td valign="top" nowrap><b>StreamRecorder</b></td><td>Record signals into <code>fif</code> format, the standard used by the <a href="https://mne.tools/">MNE</a> EEG analysis library.</td></tr>
<tr><td valign="top" nowrap><b>StreamPlayer</b></td><td>Replay recorded signals in real time, as if streamed from a live acquisition server.</td></tr>
<tr><td valign="top" nowrap><b>Decoder</b></td><td>Decoder and trainer modules. Supports LDA, regularized LDA, Random Forests, and Gradient Boosting out of the box; custom (e.g. neural-network) decoders can be added.</td></tr>
<tr><td valign="top" nowrap><b>Protocols</b></td><td>Basic protocols for training and testing. Google Glass visual feedback is supported over USB.</td></tr>
<tr><td valign="top" nowrap><b>Triggers</b></td><td>Mark event (stimulus) timings during recording. Includes common trigger event-definition files.</td></tr>
<tr><td valign="top" nowrap><b>Utils</b></td><td>Various utilities.</td></tr>
</tbody>
</table>

## Requirements

- Python **3.8+** ([Anaconda](https://www.anaconda.com/download) is recommended for an easy environment setup)
- numpy, scipy, scikit-learn
- [MNE](https://mne.tools/) 1.10+
- matplotlib, pyqtgraph, PyQt5
- pylsl
- opencv-python, pyserial, psutil
- xgboost, lightgbm

All of these are installed automatically with the package (see below).

**Optional but strongly recommended:** [OpenVibe](http://openvibe.inria.fr/downloads).
OpenVibe supports a wide range of acquisition servers, all of which become available to Neurodecode through LSL. When you run the acquisition server, tick **"LSL_EnableLSLOutput"** in Preferences so the data is streamed over the LSL network for Neurodecode to receive.

## Installation

Create a fresh virtual environment (Python 3.8 or higher):

```bash
conda create --name nd python=3.10
conda activate nd
```

Install the latest release from PyPI:

```bash
pip install neurodecode
```

Or install the development version from source:

```bash
git clone https://github.com/dbdq/neurodecode.git
cd neurodecode
pip install --editable .
```

Add the `scripts` directory to your `PATH` for convenient access to the commonly used command-line tools.

<details>
<summary><b>Troubleshooting: PyQt version conflicts</b></summary>

The Qt API is sensitive to versioning and must be compatible with all dependencies. If pyqtgraph complains about an incompatible PyQt version (e.g. PyQt < 5.12), try:

```bash
conda remove pyqt
pip install -U PyQt5
```

This is usually caused by Anaconda not shipping the latest PyQt version.
</details>

<details>
<summary><b>Windows: increase timer resolution</b></summary>

The default timer resolution on some Windows versions is 16 ms, which limits timing precision. Run the following tool and set the resolution to 1 ms or lower:
[Windows System Timer Tool](https://vvvv.org/contribution/windows-system-timer-tool)
</details>

<details>
<summary><b>Hardware triggering without a legacy parallel port</b></summary>

We developed an Arduino-based triggering system to send triggers to a parallel port over standard USB, achieving sub-millisecond extra latency compared to a physical parallel port (150 ± 25 µs). Oscilloscope results are in the `doc` folder.

```bash
git clone https://github.com/dbdq/arduino-trigger.git
```

The customized firmware should be flashed to an Arduino Micro, and the circuit design in the document folder printed to a circuit board.
</details>

<details>
<summary><b>g.USBamp users</b></summary>

Use the following customized acquisition server instead of the default LSL app, so the trigger channel is streamed as part of the signal channels:

```bash
git clone https://github.com/dbdq/gUSBamp_pycnbi.git
```

The default gUSBamp LSL server streams the event channel as a separate server rather than as part of the signal stream; the customized version supports simultaneous signal + event streaming.
</details>

<details>
<summary><b>AntNeuro eego users</b></summary>

Use the OpenVibe acquisition server and enable **"LSL output"** in preferences. If "eego" does not appear in the device list, it is likely that the additional drivers were not installed with OpenVibe.
</details>

## Running the examples

Copy the `/sample` folder to a local folder and `cd` into it.

### 1. Play data

Replay a pre-recorded EEG sample in real time, as if acquiring live signals, with a chunk size of 8. The sample was recorded from a 24-channel EEG system while a participant performed left- and right-hand motor imagery. Hardware events recorded during the experiment are also streamed over LSL.

```bash
nd_stream_player mi_left_right.fif 8
```

Setting up an LSL server and streaming the recorded data:

![Stream player](https://user-images.githubusercontent.com/6797783/199510832-c10b7df9-193b-4396-a671-15f6f8df0226.png)

### 2. Record data

Simulate real-time decoding from the brain. Data is streamed with `nd_stream_player` (above) while the receiver stays source-agnostic, enabling a full simulation of replay and validation of the processing pipeline. You can skip this step by creating a `./fif/` folder and copying the sample fif file into it.

```bash
nd_stream_recorder $PWD   # Linux
nd_stream_recorder %CD%   # Windows
```

![Stream recorder](https://user-images.githubusercontent.com/6797783/199511174-abb1ac03-eadc-488d-833a-6e303a93e331.png)

### 3. Real-time signal visualisation

Choose StreamPlayer from the list. Cursor keys change the amplitude and time scaling.

```bash
nd_stream_viewer
```

![Stream viewer](https://user-images.githubusercontent.com/6797783/199509891-a0f30cfd-c589-4004-89f0-c71ff08b4071.png)

### 4. Run an offline protocol for training

Runs an offline training protocol. This step is illustrative and can be skipped.

```bash
nd_train_mi ./config_offline.py
```

![Offline protocol](https://user-images.githubusercontent.com/6797783/199511602-6bec54d0-50dd-485c-8d3e-6fa7621cc773.png)

### 5. Train a decoder

Train a decoder from the fif file using defined events — here, left (event 11) vs. right (event 9) hand motor imagery. Events are defined in `mi_left_right_events.ini`.

```bash
nd_trainer ./config_trainer.py
```

### 6. Run an online protocol for testing

The sample is set to a 60-second time-out with no early termination, so you can watch the decoder output shift toward left or right as event 11 (left) or event 9 (right) is emitted from the stream player terminal. Other events, such as rest, are undefined and produce random behaviour.

```bash
nd_test_mi config_online.py
```

Bar position during the protocol:

![Online protocol](https://user-images.githubusercontent.com/6797783/199517521-de33e4f2-92bf-421f-8afc-9eee5c899a04.png)

Decoder output with probabilities and the corresponding bar position (accumulated probabilities):

![Decoder output](https://user-images.githubusercontent.com/6797783/199518166-67f8a4ea-dde9-4544-b95d-80ed5f0526aa.png)

Events emitted from the stream player:

![Events](https://user-images.githubusercontent.com/6797783/199514155-a94bbb71-c2dc-43d5-81e8-2bd4916a05e4.png)

There is still plenty of room to optimize speed across the codebase, and contributions are very welcome. For comments or feedback, contact lee.kyuh@gmail.com.

## License

Released under the [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0.en.html).

## Citation

If any of this code was useful for your project, please consider citing below paper that conceived this project:
- Kyuhwa Lee *et al.*, "A Brain-Controlled Exoskeleton with Cascaded Event-Related Desynchronization Classifiers," *Robotics and Autonomous Systems*, Elsevier, 2016, p. 15–23.
