# Introduction

Neurodecode provides a real-time brain signal decoding framework with a modular software design. The decoding performance was recognised at [Microsoft Brain Signal Decoding competition](https://github.com/dbdq/microsoft_decoding) with the <i>First Prize Award</i> (2016) for high decoding accuracy (2nd out of 1863 algorithms). It has been applied on a couple of online decoding projects with various electrode types including EEG, ECoG, DBS, and microelectrode arrays, and on several acquisition systems including AntNeuro eego, g.tec gUSBamp, BioSemi ActiveTwo, BrainProducts actiCHamp and Wearable Sensing. The decoding runs at approximately 15 classifications per second(cps) on a 4th-gen i7 laptop with 64-channel setup at 512 Hz sampling rate. High-speed decoding up to 200 cps was achieved using process-interleaving technique on 8 cores. It has been tested on both Linux and Windows using Python 3.7.

The underlying data communication is based on Lab Streaming Layer (LSL) which provides sub-millisecond time synchronization accuracy. Any signal acquisition system supported by native LSL or OpenVibe is also supported by Neurodecode. Since the data communication is based on TCP, signals can be also transmitted wirelessly. For more information about LSL, please visit:
[https://github.com/sccn/labstreaminglayer](https://github.com/sccn/labstreaminglayer)


# Important modules

### StreamReceiver
The base module for acquiring signals used by other modules such as Decoder, StreamViewer and StreamRecorder.

### StreamViewer
Visualize signals in real time with spectral filtering, common average filtering options and real-time FFT.

### StreamRecorder
Record signals into fif format, a standard format mainly used in [MNE EEG analysis library](http://martinos.org/mne/).

### StreamPlayer
Replay the recorded signals in real time as if it was transmitted from a real acquisition server.

### Decoder
This folder contains decoder and trainer modules. Currently, LDA, regularized LDA, Random Forests, and Gradient Boosting Machines are supported as the standard classifier type. Neural network-based decoders can be added as a customer decoder.

### Protocols
Contains some basic protocols for training and testing. Google Glass visual feedback is supported through USB communication.

### Triggers
Triggers are used to mark event (stimulus) timings during the recording. This folder contains common trigger event definition files. 

### Utils
Contains various utilities.


# Prerequisites

Anaconda is recommended for easy installation of Python environment.

Neurodecode depends on following packages:
  - scipy
  - numpy
  - PyQt5
  - scikit-learn
  - pylsl
  - mne 0.14 or later
  - matplotlib 2.1.0 or later
  - pyqtgraph
  - opencv-python
  - pyserial
  - future
  - configparser
  - xgboost
  - psutil

Optional but strongly recommended:
  - [OpenVibe](http://openvibe.inria.fr/downloads)

OpenVibe supports a wide range of acquisition servers and all acquisition systems supported by OpenVibe are supported by Neurodecode through LSL. Make sure you tick the checkbox "LSL_EnableLSLOutput" in Preferences when you run acquisition server. This will stream the data through the LSL network from which Neurodecode receives data. 

# Installation

Neurodecode can be installed from PyPI.
```
pip install neurodecode
```

To install the latest version, clone the repository and run setup script:
```
git clone https://github.com/dbdq/neurodecode.git
pip install --editable .
```
Add "scripts" directory to PATH environment variable for convenient access to commonly used scripts.

### PyQt version problem
The Qt API is very sensitive to version and needs to be compatible with all dependencies.
If you experience pyqtgraph complaining incompatible PyQt version (e.g. PyQt < 5.12), try:
```
conda remove pyqt
pip install -U PyQt5
```
This can be caused by Anaconda not having the latest PyQt version.


### For Windows users, increase timer resolution
The default timer resolution in some Windows versions is 16 ms, which can limit the precision of timings. It is recommended to run the following tool and set the resolution to 1 ms or lower:
[https://vvvv.org/contribution/windows-system-timer-tool](https://vvvv.org/contribution/windows-system-timer-tool)


### Hardware triggering without legacy parallel port
We have also developed an Arduino-based triggering system as we wanted to send triggers to a parallel port using standard USB ports. We achieved sub-millisecond extra latency compared to physical parallel port (150 +- 25 us). Experimental results using oscilloscope can be found in "doc" folder. The package can be downloaded by:
```
git clone https://github.com/dbdq/arduino-trigger.git
```
The customized firmware should be installed on Arduino Micro and the circuit design included in the document folder should be printed to a circuit board.


### For g.USBamp users
The following customized acquisition server is needed instead of default LSL app to receive the trigger channel as part of signal streaming channels:
```
git clone https://github.com/dbdq/gUSBamp_pycnbi.git
```
because the default gUSBamp LSL server do not stream event channel as part of the signal stream but as a separate server. The customized version supports simultaneous signal+event channel streaming. 


### For AntNeuro eego users
Use the OpenVibe acquisition server and make sure to check "LSL output" in preference.  If you don't see "eego" from the device selection, it's probably because you didn't install the additional drivers when you installed OpenVibe.

<br>
<br>

# Running examples

To run this example, copy the /sample folder to your local folder and cd into it.

### 1. Play data
Replay a pre-recorded EEG recording sample in real-time as if acquiring signals from brain with a chunk size of 8.
The sample data was recorded using a 24-channel EEG system from a participant doing a left and right hand motor imagery.
The hardware events recorded during the experiment is also streamed via LSL network.

```nd_stream_player mi_left_right.fif 8```

Screenshot of setting up an LSL server and streaming the recorded data.
![image](https://user-images.githubusercontent.com/6797783/199510832-c10b7df9-193b-4396-a671-15f6f8df0226.png)

### 2. Record data  
Simulate real-time decoding from the brain. We are streaming the data using nd_stream_player script above while
the receiver is source-agnostic, allowing the full simulation of replaying and the validation of data processing.  
This step can be skipped if you create a folder ./fif/ and copy the sample fif file into ./fif/.

```
nd_stream_recorder $PWD # for Linux
nd_stream_recorder %CD% # for Windows
```
![image](https://user-images.githubusercontent.com/6797783/199511174-abb1ac03-eadc-488d-833a-6e303a93e331.png)

### 3. Real-time signal visualisation (choose StreamPlayer from the list)  
```nd_stream_viewer```

Sample visualisation of streamed signals. Cursor keys allow different amplitude and time scalings.
![image](https://user-images.githubusercontent.com/6797783/199509891-a0f30cfd-c589-4004-89f0-c71ff08b4071.png)

### 4. Run an offline protocol for training
Runs an offline training protocol. This step is just for explanation purpose and can be skipped.  
```nd_train_mi ./config_offline.py```

Snapshot of the offline protocol.  
![image](https://user-images.githubusercontent.com/6797783/199511602-6bec54d0-50dd-485c-8d3e-6fa7621cc773.png)

### 5. Train a decoder  
Train a decoder using the fif file with defined events. In this example, it is left (event 11) vs right (event 9) hand motor imagery.  
Events are defined in mi_left_right_events.ini file.

```nd_trainer ./config_trainer.py```

### 6. Run an online protocol for testing
The provided sample is set to 60 seconds time-out without early termination so you can see
the decoder output changes to left or right when the left (event 11) or right(event 9) is emitted 
from the stream player terminal. Other events such as rest is undefined and will behave in random direction.

```nd_test_mi config_online.py```

Snapshot of the protocol showing the bar position.  
![image](https://user-images.githubusercontent.com/6797783/199517521-de33e4f2-92bf-421f-8afc-9eee5c899a04.png)  

Sample decoder output with probabilities and the corresponding bar position, which represents the accumulated probabilities.  
![image](https://user-images.githubusercontent.com/6797783/199518166-67f8a4ea-dde9-4544-b95d-80ed5f0526aa.png)  

Example of events being emitted from the stream player.  
![image](https://user-images.githubusercontent.com/6797783/199514155-a94bbb71-c2dc-43d5-81e8-2bd4916a05e4.png)  

There are still plenty of possibilities to optimize the speed in many parts of the code. Any contribution is welcome. Please contact lee.kyuh@gmail.com for any comment / feedback.


# Copyright and license
The codes are released under [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0.en.html).


# Citation
This package was developed as part of the following works:
  - \*Yohann Thenaisie & \*Kyuhwa Lee <i>et al.</i>, "Principles of gait encoding in the subthalamic nucleus of people with Parkinson's disease", <i>Science Translational Medicine</i>, 2022, Vol. 14, No. 661, p. eabo1800.<br>(* co-first authors)
  - Kyuhwa Lee <i>et al.</i>, "A Brain-Controlled Exoskeleton with Cascaded Event-Related Desynchronization Classifiers", <i>Robotics and Autonomous Systems</i>, Elsevier, 2016, p. 15-23.
   
If some of the code here were useful or helpful for your project, I would greatly appreciate if you could cite one of the above papers.
