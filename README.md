# Introduction

Neurodecode provides a real-time brain signal decoding framework. The decoding performance was recognised at [Microsoft Brain Signal Decoding competition](https://github.com/dbdq/microsoft_decoding) with the <i>First Prize Award</i> (2016) for high decoding accuracy (2nd out of 1863 algorithms). It has been applied on a couple of online decoding projects with various electrode types including EEG, ECoG and DBS, and on several acquisition systems including AntNeuro eego, g.tec gUSBamp, BioSemi ActiveTwo, BrainProducts actiCHamp and Wearable Sensing. With modular software design, the decoding runs at approximately 15 classifications per second(cps) on a 4th-gen i7 laptop with 64-channel setup at 512 Hz sampling rate. High-speed decoding up to 200 cps was achieved using process-interleaving technique on 8 cores. It has been tested on both Linux and Windows using Python 3.7.

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
This folder contains decoder and trainer modules. Currently, LDA, regularized LDA, Random Forests, and Gradient Boosting Machines are supported as the classifier type. Neural Network-based decoders are currently under experiment.

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
python setup.py develop
```
Add "scripts" directory to PATH environment variable for convenient access to commonly used scripts.

## PyQt version problem
The Qt library is very sensitive to version and needs to be compatible with all dependencies.
If you experience pyqtgraph complaining incompatible PyQt version (e.g. PyQt < 5.12), try:
```
conda remove pyqt
pip install -U PyQt5
```
This can be caused by Anaconda not having the latest PyQt version.


## For Windows users, increase timer resolution
The default timer resolution in some Windows versions is 16 ms, which can limit the precision of timings. It is recommended to run the following tool and set the resolution to 1 ms or lower:
[https://vvvv.org/contribution/windows-system-timer-tool](https://vvvv.org/contribution/windows-system-timer-tool)


## Hardware triggering without legacy parallel port
We have also developed an Arduino-based triggering system as we wanted to send triggers to a parallel port using standard USB ports. We achieved sub-millisecond extra latency compared to physical parallel port (150 +- 25 us). Experimental results using oscilloscope can be found in "doc" folder. The package can be downloaded by:
```
git clone https://github.com/dbdq/arduino-trigger.git
```
The customized firmware should be installed on Arduino Micro and the circuit design included in the document folder should be printed to a circuit board.


## For g.USBamp users
The following customized acquisition server is needed instead of default LSL app to receive the trigger channel as part of signal streaming channels:
```
git clone https://github.com/dbdq/gUSBamp_pycnbi.git
```
because the default gUSBamp LSL server do not stream event channel as part of the signal stream but as a separate server. The customized version supports simultaneous signal+event channel streaming. 


## For AntNeuro eego users
Use the OpenVibe acquisition server and make sure to check "LSL output" in preference.  If you don't see "eego" from the device selection, it's probably because you didn't install the additional drivers when you installed OpenVibe.



# Running examples

To run this example, copy the sample data and codes to a new folder and cd into this folder.

1. Replay data in real-time as if acquiring signals from brain with a chunk size of 8  
```nd_stream_player mi_left_right.fif 8```

2. Record streaming data  
```
nd_stream_recorder $PWD/records # for Linux
nd_stream_recorder %CD%\records # for Windows
```

3. Visualise the signals (choose StreamPlayer from the list)  
```nd_stream_viewer```

4. Run an offline protocol for training (can be skipped for this exercise)  
```nd_train_mi config_offline.py```

5. Train a decoder  
```nd_trainer config_trainer.py```

6. Run an online protocol and test a decoder  
```nd_test_mi config_online.py```

There are still plenty of possibilities to optimize the speed in many parts of the code. Any contribution is welcome. Please contact lee.kyuh@gmail.com for any comment / feedback.


# Copyright and license
The codes are released under [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0.en.html).


# Citation
This package was developed as part of the following works:
  - Yohann Thenaisie* and Kyuhwa Lee* <i>et al.</i>, "Principles of gait encoding in the subthalamic nucleus of people with Parkinson's disease", <i>Science Translational Medicine</i>, 2022, Vol. 14, No. 661, p. eabo1800.<br>(* first authors as equal contribution)
  - Kyuhwa Lee <i>et al.</i>, "A Brain-Controlled Exoskeleton with Cascaded Event-Related Desynchronization Classifiers", <i>Robotics and Autonomous Systems</i>, Elsevier, 2016, p. 15-23.
   
If some of the code here were useful or helpful for your project, I would greatly appreciate if you could cite one of the above papers.
