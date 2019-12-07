from __future__ import print_function, division

"""
Export fif data into mat files.

"""

import os
import mne
import scipy.io
import pycnbi.utils.q_common as qc
import pycnbi.utils.pycnbi_utils as pu
from pycnbi import logger

def fif2mat_file(fif_file, out_dir='./'):
    raw, events = pu.load_raw(fif_file)
    events[:,0] += 1 # MATLAB uses 1-based indexing
    sfreq = raw.info['sfreq']
    data = dict(signals=raw._data, events=events, sfreq=sfreq, ch_names=raw.ch_names)
    fname = qc.parse_path(fif_file).name
    matfile = '%s/%s.mat' % (out_dir, fname)
    scipy.io.savemat(matfile, data)
    logger.info('Exported to %s' % matfile)

def fif2mat(input_path):
    if os.path.isdir(input_path):
        out_dir = '%s/mat_files' % input_path
        qc.make_dirs(out_dir)
        num_processed = 0
        for rawfile in qc.get_file_list(input_path, fullpath=True):
            if rawfile[-4:] != '.fif':
                continue
            fif2mat_file(rawfile, out_dir)
            num_processed += 1
        if num_processed == 0:
            logger.warning('No fif files found in the path.')
    elif os.path.isfile(input_path):
        out_dir = '%s/mat_files' % qc.parse_path(input_path).dir
        qc.make_dirs(out_dir)
        fif2mat_file(input_path, out_dir)
    else:
        raise ValueError('Neither directory nor file: %s' % input_path)
    logger.info('Finished.')

if __name__ == '__main__':
    # path to fif file(s)
    input_path = r'D:\data\Records\fif'
    fif2mat(input_path)
