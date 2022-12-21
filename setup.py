from setuptools import find_packages, setup

# backward compatibility for old pycnbi users
'''
# symlink pycnbi to neurodecode
import os
import sys
if not os.path.exists('./pycnbi'):
    if sys.platform.startswith('win'):
        os.system('mklink pycnbi neurodecode /J')
    else:
        os.symlink('./neurodecode', './pycnbi', True)
'''

setup(
    name='neurodecode',
    version='2.0.1',
    author='Kyuhwa Lee, Arnaud Desvachez',
    author_email='lee.kyuh@gmail.com, arnaud.desvachez@gmail.com',
    license='The GNU General Public License',
    url='https://github.com/dbdq/neurodecode/',
    description='Real-time brain signal decoding framework',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    package_data={'neurodecode.stream_viewer':['*.ini']},
    install_requires=[
        'h5py>=2.7',
        'opencv_python>=3.4',
        'numpy>=1.16',
        'scipy>=1.2',
        'colorama>=0.3.9',
        'xgboost>=0.81',
        'matplotlib>=3.0.2',
        'mne>=0.16',
        'psutil>=5.4.8',
        'setuptools>=39.0.1',
        'pyqtgraph>=0.13',
        'pylsl>=1.12.2',
        'ipython>=6',
        'PyQt5>=5',
        'pyxdf>=1.15.2',
        'pyserial>=3.4',
        'simplejson>=3.16.0',
        'scikit_learn>=0.21',
        'future',
        'configparser',
        'lightgbm>=2.3',
        'mat73'
    ],
    entry_points = {'console_scripts':[
        'nd_stream_viewer=neurodecode.stream_viewer.stream_viewer:main',
        'nd_fif_info=neurodecode.utils.fif_info:main',
        'nd_fif_resample=neurodecode.utils.fif_resample:main',
        'nd_fif2mat=neurodecode.utils.fif2mat:main',
        'nd_add_lsl_events=neurodecode.utils.add_lsl_events:main',
        'nd_parse_features=neurodecode.utils.parse_features:main',
        'nd_convert2fif=neurodecode.utils.convert2fif:main',
        'nd_train_mi=neurodecode.protocols.mi.train_mi:main',
        'nd_test_mi=neurodecode.protocols.mi.test_mi:main',
        'nd_stream_player=neurodecode.stream_player.stream_player:main',
        'nd_stream_recorder=neurodecode.stream_recorder.stream_recorder:main',
        'nd_tfr_export=neurodecode.analysis.tfr_export:main',
        'nd_tfr_export_each_file=neurodecode.analysis.tfr_export_each_file:main',
        'nd_trainer=neurodecode.decoder.trainer:main',
        'nd_decoder=neurodecode.decoder.decoder:main'
    ]}
)
