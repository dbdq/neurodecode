"""
Real-time PSD visualization script

TODO: make it a function or a class

Author:
Kyuhwa Lee

"""

from mne.decoding import PSDEstimator
from neurodecode.stream_receiver.stream_receiver import StreamReceiver
import neurodecode.utils.pycnbi_utils as pu
import sklearn
import numpy as np
import cv2
import mne
import os
mne.set_log_level('ERROR')
os.environ['OMP_NUM_THREADS'] = '1' # actually improves performance for multitaper


def get_psd(sr, psde, picks):
    sr.acquire()
    w, ts = sr.get_window()  # w = times x channels
    w = w.T  # -> channels x times

    # apply filters. Important: maintain the original channel order at this point.
    w = pu.preprocess(w, sfreq=sfreq, spatial=spatial, spatial_ch=spatial_ch,
        spectral=spectral, spectral_ch=spectral_ch, notch=notch,
        notch_ch=notch_ch, multiplier=multiplier)

    # select the same channels used for training
    w = w[picks]

    # psde.transform = [channels x freqs]
    psd = psde.transform(w)

    return psd

if __name__ == '__main__':
    # LSL setting
    amp_name = 'StreamPlayer'
    amp_serial = None

    # define channels to show
    channel_picks = ['Cz', 'C3', 'C4', 'Pz', 'P3', 'P4', 'O1', 'O2']

    # PSD
    w_seconds = 1
    fmin = 1
    fmax = 40

    # filters
    spatial = 'car'
    spatial_ch = None
    spectral = None
    spectral_ch = None
    notch = None
    notch_ch = None
    multiplier = 1

    # viz settings
    mul_x = 50
    mul_y = 20
    fq_offset = 55
    ch_offset = 30
    screen_offset_x = 100
    screen_offset_y = 100

    # start
    sr = StreamReceiver(window_size=w_seconds, amp_name=amp_name, amp_serial=amp_serial)
    sfreq = sr.sample_rate
    psde = PSDEstimator(sfreq=sfreq, fmin=fmin, fmax=fmax, bandwidth=None, adaptive=False,
        low_bias=True, n_jobs=1, normalization='length', verbose=None)
    ch_names = sr.get_channel_names()
    fq_res = 1 / w_seconds
    hz_list = []
    f = fmin
    while f <= fmax:
        hz_list.append(f)
        f += fq_res
    picks = [ch_names.index(ch) for ch in channel_picks]
    psd = get_psd(sr, psde, picks).T # freq x ch
    assert len(hz_list) == psd.shape[0], (len(hz_list), psd.shape[0])
    cv2.namedWindow("img", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("img", screen_offset_x, screen_offset_y)
    #cv2.setWindowProperty("img", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
    img_x = psd.shape[1] * mul_x
    img_y = psd.shape[0] * mul_y
    img = np.zeros((img_y + ch_offset, img_x + fq_offset), np.uint8)

    # channels
    for x in range(psd.shape[1]):
        cv2.putText(img, '%s' % ch_names[picks[x]], (x * mul_x + fq_offset + 13, img_y + ch_offset - 10),
            cv2.FONT_HERSHEY_DUPLEX, 0.5, [255,255,255], 1, cv2.LINE_AA)
    # frequencies
    for y in range(psd.shape[0]):
        cv2.putText(img, '%5.0f' % hz_list[y], (1, y * mul_y + 15), cv2.FONT_HERSHEY_DUPLEX,
            0.5, [255,255,255], 1, cv2.LINE_AA)
    cv2.putText(img, 'Hz/Ch', (1, img_y + ch_offset - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, [255,255,255], 1, cv2.LINE_AA)

    scaler = sklearn.preprocessing.MinMaxScaler((0, 255))
    key = 0
    while key != 27:
        psd = get_psd(sr, psde, picks).T
        psd_log = scaler.fit_transform(np.log10(psd))
        psd_img = cv2.resize(psd_log, (img_x, img_y), interpolation=0)
        img[:-ch_offset, fq_offset:] = psd_img
        cv2.imshow("img", img)
        key = cv2.waitKey(1)
