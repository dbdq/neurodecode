from __future__ import print_function, division

"""
Useful Python utilities

Author: Kyuhwa Lee
Imperial College of Science, Technology and Medicine

"""

# set Q_VERBOSE= 0 to make it silent. 1:verbose, 2:extra verbose
Q_VERBOSE = 0


import os
import sys
import pdb
import code
import time
import math
import mat73
import scipy.io
import shutil
import logging
import inspect
import traceback
import itertools
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt

# logger
try:
    from neurodecode import logger
except ImportError:
    logging.basicConfig(format='%(msg)s', level='INFO')
    logger = logging.getLogger(__name__)
    logger.propagate = False

# pickle
try:
    import cPickle as pickle  # Python 2 (cPickle = C version of pickle)
except ImportError:
    import pickle  # Python 3 (C version is the default)


'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
 Math
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_array(x):
    return 1 / (1 + np.exp(-x))

def dirichlet(n):
    """
    Uniform Dirichlet distribution with sigma(alpha)=1.0
    """
    alpha = 1.0 / n
    return 1 / beta(alpha, n) * ((1 / n) ** (alpha - 1)) ** n

def beta(alpha, n):
    """
    Multinomial Beta function with uniform alpha values

    n: number of rule probabilities
    """
    return math.gamma(alpha) ** n / math.gamma(n * alpha)

def poisson(mean, k):
    """
    Poisson distribution. We use k-1 since the minimum length is 1, not 0.
    """
    return (mean ** (k - 1) * math.exp(-mean)) / math.factorial(k - 1)

def average_every_n(arr, n):
    """
    Average every n elements of a numpy array

    if not len(arr) % n == 0, it will be trimmed to the closest divisible length
    """
    end = n * int(len(arr) / n)
    return np.mean(arr[:end].reshape(-1, n), 1)

def zscore_mod(X, axis=0):
    """
    Modified z-score for multidimensional arrays

    MAD = median(abs(X-median(X)))
    MeanAD = mean(abs(X-mean(X)))
    If MAD does equal 0 (if at least 50% of samples have the same value)
      Subtract the median from the score and divide by 1.253314*MeanAD.
      1.253314*MeanAD approximately equals the standard deviation.
    If MAD does not equal 0
      Subtract the median from the score and divide by 1.486*MAD.
      1.486*MAD approximately equals the standard deviation.

    Reference:
    https://www.ibm.com/docs/en/cognos-analytics/11.1.0?topic=terms-modified-z-score
    """
    # medians are computed for each trial for each feature
    X_med = np.median(X, axis=axis)
    # expand dimension for broadcasting
    X_med_diff = X - np.expand_dims(X_med, axis=axis)
    # compute MAD
    X_mad = np.median(np.abs(X_med_diff), axis=axis)

    if X_mad.all():
        # if all MADs are non-zero
        X_norm = X_med_diff / (1.486 * np.expand_dims(X_mad, axis))
    else:
        # compute MeanAD
        X_mean = np.mean(X, axis=axis)
        X_mean_ad = np.mean(abs(X - np.expand_dims(X_mean, axis=axis)), axis=axis)
        X_mad_corrected = X_mad * 1.486
        # replace 0 in MAD with the corresponding meanAD value
        for zero_loc in np.array(np.where(X_mad==0)).T:
            zero_loc = tuple(zero_loc) # every index containing 0 in MAD
            X_mad_corrected[zero_loc] = 1.253314 * X_mean_ad[zero_loc]
        X_norm = X_med_diff / np.expand_dims(X_mad_corrected, axis)
    return X_norm


'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
 List/Dict related
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''

def list2string(vec, fmt, sep=' '):
    """
    Convert a list to string with formatting, separated by sep (default is space).
    Example: fmt= '%.32e', '%.6f', etc.
    """
    return sep.join((fmt % x for x in vec))

def flatten_list(l):
    return list(itertools.chain.from_iterable(l))

def get_index_min(seq):
    """
    Get the index of the minimum item in a list or dict
    """
    if type(seq) == list:
        return min(range(len(seq)), key=seq.__getitem__)
    elif type(seq) == dict:
        return min(seq, key=seq.__getitem__)
    else:
        logger.error('Unsupported input %s' % type(seq))
        return None

def get_index_max(seq):
    """
    Get the index of the maximum item in a list or dict
    """
    if type(seq) == list:
        return max(range(len(seq)), key=seq.__getitem__)
    elif type(seq) == dict:
        return max(seq, key=seq.__getitem__)
    else:
        logger.error('Unsupported input %s' % type(seq))
        return None

def sort_by_value(s, rev=False):
    """
    Sort dictionary or list by value and return a sorted list of keys and values.
    Values must be hashable and unique.
    """
    assert type(s) == dict or type(s) == list, 'Input must be a dictionary or list.'
    if type(s) == list:
        s = dict(enumerate(s))
    s_rev = dict((v, k) for k, v in s.items())
    if Q_VERBOSE > 0 and not len(s_rev) == len(s):
        logger.warning('sort_by_value(): %d identical values' % (len(s.values()) - len(set(s.values())) + 1))
    values = sorted(s_rev, reverse=rev)
    keys = [s_rev[x] for x in values]
    return keys, values


'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
 File I/O
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''

def get_file_list(path, fullpath=True, recursive=False):
    """
    Get files with or without full path.
    """
    path = path.replace('\\', '/')
    if not path[-1] == '/': path += '/'

    if recursive == False:
        if fullpath == True:
            filelist = [path + f for f in os.listdir(path) if os.path.isfile(path + '/' + f) and f[0] != '.']
        else:
            filelist = [f for f in os.listdir(path) if os.path.isfile(path + '/' + f) and f[0] != '.']
    else:
        filelist = []
        for root, dirs, files in os.walk(path):
            root = root.replace('\\', '/')
            if fullpath == True:
                [filelist.append(root + '/' + f) for f in files]
            else:
                [filelist.append(f) for f in files]
    return sorted(filelist)

def get_dir_list(path, recursive=False, no_child=False):
    """
    Get directory list relative to path.

    Input:
        recusrive: search recursively if True.
        no_child: search directories having no child directory (leaf nodes)
    """
    path = path.replace('\\', '/')
    if not path[-1] == '/': path += '/'

    if recursive == True:
        pathlist = []
        for root, dirs, files in os.walk(path):
            root = root.replace('\\', '/')
            [pathlist.append(root + '/' + d) for d in dirs]

            if no_child:
                for p in pathlist:
                    if len(get_dir_list(p)) > 0:
                        pathlist.remove(p)

    else:
        pathlist = [path + f for f in os.listdir(path) if os.path.isdir(path + '/' + f)]
        if no_child:
            for p in pathlist:
                if len(get_dir_list(p)) > 0:
                    pathlist.remove(p)

    return sorted(pathlist)

def make_dirs(dirname, delete=False):
    """
    Recusively create directories.
    if delete=true, directory will be deleted first if exists.
    """
    if os.path.exists(dirname) and delete == True:
        try:
            shutil.rmtree(dirname)
        except OSError:
            logger.error('Directory was not completely removed. (Perhaps a Dropbox folder?). Continuing.')
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def save_obj(fname, obj, protocol=4):
    """
    Save python object into a file
    Set protocol=2 for Python 2 compatibility
    """
    with open(fname, 'wb') as fout:
        pickle.dump(obj, fout, protocol)

def load_obj(fname):
    """
    Read python object from a file
    """
    try:
        with open(fname, 'rb') as f:
            return pickle.load(f)
    except UnicodeDecodeError:
        # usually happens when trying to load Python 2 pickle object from Python 3
        with open(fname, 'rb') as f:
            return pickle.load(f, encoding='latin1')
    except:
        msg = 'load_obj(): Cannot load pickled object file "%s". The error was:\n%s\n%s' %\
              (fname, sys.exc_info()[0], sys.exc_info()[1])
        raise IOError(msg)

def loadtxt_fast(filename, delimiter=',', skiprows=0, dtype=float):
    """
    Much faster matrix loading than numpy's loadtxt
    http://stackoverflow.com/a/8964779
    """
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        loadtxt_fast.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, loadtxt_fast.rowlength))
    return data

def parse_path(file_path):
    """
    Input:
        full path

    Returns:
        self.dir = base directory of the file
        self.name = file name without extension
        self.ext = file extension
    """
    class path_info:
        def __init__(self, path):
            path_abs = os.path.realpath(path).replace('\\', '/')
            s = path_abs.split('/')
            f = s[-1].split('.')
            basedir = '/'.join(s[:-1])
            if len(f) == 1:
                name, ext = f[-1], ''
            else:
                name, ext = '.'.join(f[:-1]), f[-1]
            self.dir = basedir
            self.name = name
            self.ext = ext
            self.txt = 'self.dir=%s\nself.name=%s\nself.ext=%s\n' % (self.dir, self.name, self.ext)
        def __repr__(self):
            return self.txt
        def __str__(self):
            return self.txt

    return path_info(file_path)

def parse_path_list(path):
    """
    Input:
        full path

    Returns:
        base dir, file(or dir) name, extension (if file)
    """

    path_abs = os.path.realpath(path).replace('\\', '/')
    s = path_abs.split('/')
    f = s[-1].split('.')
    basedir = '/'.join(s[:-1]) + '/'
    if len(f) == 1:
        name, ext = f[-1], ''
    else:
        name, ext = '.'.join(f[:-1]), f[-1]

    return basedir, name, ext

def forward_slashify(txt):
    return txt.replace('\\\\', '/').replace('\\', '/')


'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
 Timer class
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''

class Timer(object):
    """
    Timer class

    if autoreset=True, timer is reset after any member function call
    """
    def __init__(self, autoreset=False):
        self.autoreset = autoreset
        self.reset()

    def sec(self):
        read = time.time() - self.ref
        if self.autoreset: self.reset()
        return read

    def msec(self):
        return self.sec() * 1000.0

    def reset(self):
        self.ref = time.time()

    def sleep_atleast(self, sec):
        """
        Sleep up to sec seconds
        It's more convenient if autoreset=True
        """
        timer_sec = self.sec()
        if timer_sec < sec:
            time.sleep(sec - timer_sec)
            if self.autoreset: self.reset()


'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
 Debugging
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''

def auto_debug():
    """
    Triggers debugging mode automatically when AssertionError is raised

    Snippet from:
      stackoverflow.com/questions/242485/starting-python-debugger-automatically-on-error
    """
    def debug_info(type, value, tb):
        if hasattr(sys, 'ps1') or not sys.stderr.isatty() or type == KeyboardInterrupt:
            # interactive mode or no tty-like device
            sys.__excepthook__(type, value, tb)
        else:
            # non-interactive mode
            logger.exception()
            pdb.pm()
    sys.excepthook = debug_info

# enter interactive shell within the caller's scope
def shell():
    """
    Enter interactive shell within the caller's scope
    """
    logger.info('*** Entering interactive shell. Ctrl+D to return. ***')
    stack = inspect.stack()
    try:  # globals are first loaded, then overwritten by locals
        globals_ = {}
        globals_.update({key:value for key, value in stack[1][0].f_globals.items()})
        globals_.update({key:value for key, value in stack[1][0].f_locals.items()})
    finally:
        del stack
    code.InteractiveConsole(globals_).interact()

def run_multi(cmd_list, cores=0, quiet=False):
    """
    Input
    -----
    cmd_list: list of commands just like when you type on bash
    cores: number of cores to use (use all cores if 0)
    Logging tip: "command args > log.txt 2>&1"
    """
    if cores == 0: cores = mp.cpu_count()
    pool = mp.Pool(cores)
    processes = []
    for c in cmd_list:
        if not quiet:
            logger.info(cmd)
        processes.append(pool.apply_async(os.system, [cmd]))
    for proc in processes:
        proc.get()
    pool.close()
    pool.join()

# print_c definition: print texts in color
try:
    import colorama
    colorama.init()

    def print_c(msg, color=None, end='\n'):
        """
        Colored print using colorama.

        Fullset:
            https://pypi.python.org/pypi/colorama
            Fore: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
            Back: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
            Style: DIM, NORMAL, BRIGHT, RESET_ALL

        TODO:
            Make it using *args and **kwargs to support fully featured print().

        """
        if color is None:
            print(str(msg), end=end)
            return
        color = str(color)
        if len(color) != 1:
            raise ValueError('Color parameter must be a single color code, not %s' % type(color))
        if color.upper() == 'B':
            c = colorama.Fore.BLUE
        elif color.upper() == 'R':
            c = colorama.Fore.RED
        elif color.upper() == 'G':
            c = colorama.Fore.GREEN
        elif color.upper() == 'Y':
            c = colorama.Fore.YELLOW
        elif color.upper() == 'W':
            c = colorama.Fore.WHITE
        elif color.upper() == 'C':
            c = colorama.Fore.CYAN
        else:
            logger.error('print_c(): Unknown color code %s' % color)
            raise ValueError
        print(colorama.Style.BRIGHT + c + str(msg) + colorama.Style.RESET_ALL, end=end)

except ImportError:
    logger.warning('colorama module not found. print_c() will ignore color codes.')
    def print_c(msg, color, end='\n'):
        print(msg, end=end)


'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
 MATLAB
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''

def matlab(codes):
    """ Execute Matlab snippets """
    exe = 'matlab -nojvm -nodisplay -nosplash -wait -automation -r \"cd %s; %s; exit;\"' % (os.getcwd(), codes)
    # exe= 'matlab -nojvm -nodisplay -nosplash -wait -automation -r \"%s; exit;\"'% codes
    os.system(exe)

def loadmat(filename):
    '''
    Proper mat file loading perserving the correct structure
    https://stackoverflow.com/review/suggested-edits/21667510

    this function should be called instead of direct scipy.io.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], scipy.io.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _has_struct(elem):
        """Determine if elem is an array and if any array item is a struct"""
        return isinstance(elem, np.ndarray) and any(isinstance(
                    e, scipy.io.matlab.mio5_params.mat_struct) for e in elem)

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif _has_struct(elem):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, scipy.io.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif _has_struct(sub_elem):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    try:
        # old Matlab format
        data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
        return _check_keys(data)
    except NotImplementedError:
        # Matlab 7.3 and higher (HDF)
        return mat73.loadmat(filename)


'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
 Scikit-learn
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''

def confusion_matrix(Y_true, Y_pred, label_len=6):
    """
    Generate confusion matrix in a string format

    Input
    -----
        Y_true: true labels
        Y_pred: test labels
        label_len: maximum label text length (minimum length: 6)

    Output
    ------
        (cfmat, acc)
        cfmat: confusion matrix (string)
            X-axis: prediction
            Y-axis: ground truth
        acc: accuracy (float)
    """

    # find labels
    if type(Y_true) == np.ndarray:
        Y_labels = np.unique(Y_true)
    else:
        Y_labels = [x for x in set(Y_true)]
    if label_len < 6:
        label_len = 6
        logger.warning('label_len < 6. Setting to 6.')
    label_tpl = '%' + '-%ds' % label_len
    col_tpl = '%' + '-%d.2f' % label_len

    # sanity check
    if len(Y_pred) > len(Y_true):
        raise RuntimeError('Y_pred has more items than Y_true')
    elif len(Y_pred) < len(Y_true):
        Y_true = Y_true[:len(Y_pred)]

    cm = sklearn.metrics.confusion_matrix(Y_true, Y_pred, labels=Y_labels)

    # compute confusion matrix
    cm_rate = cm.copy().astype('float')
    # cm_rate= cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # this can have NaN
    cm_sum = np.sum(cm, axis=1)

    for r, s in zip(cm_rate, cm_sum):
        if s > 0:
            r /= s
    cm_txt = label_tpl % 'gt\dt'
    for l in Y_labels:
        cm_txt += label_tpl % l[:label_len]
    cm_txt += '\n'
    for l, r in zip(Y_labels, cm_rate):
        cm_txt += label_tpl % l[:label_len]
        for c in r:
            cm_txt += col_tpl % c
        cm_txt += '\n'

    # compute accuracy
    correct = 0.0
    for c in range(cm.shape[0]):
        correct += cm[c][c]
    cm_sum = cm.sum()
    if cm_sum > 0:
        acc = correct / cm.sum()
    else:
        acc = 0.0

    return cm_txt, acc


'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
 Matplotlib
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''

def plot_imagesc(img, x=None, y=None, title='', colorbar=True, **kwargs):
    """
    Plot 2-D array as image with autoscaling with a color range [med-std, med+std]
    """
    # sanity check
    if (x is not None and y is not None) and (len(x) != img.shape[1] or len(y) != img.shape[0]):
        raise ValueError('yicks (%d) and xticks (%d) do not correspond to image shape %s' % (len(x), len(y), img.shape))

    plt.imshow(img, aspect='auto', interpolation='none', origin='lower', **kwargs)
    if colorbar:
        plt.colorbar()
    std = np.std(img, axis=None)
    med = np.median(img, axis=None)
    plt.clim([med-std, med+std])
    if x is not None:
        if len(x) <= 10:
            xstep = 1
        else:
            xstep = len(x) // 10
        if type(x[0]) in [str, np.str_]:
            xticklabels = x[::xstep]
        else:
            xticklabels = x.round(6)[::xstep]
        plt.xticks(range(0, img.shape[1], xstep), xticklabels)
    if y is not None:
        if len(y) <= 10:
            ystep = 1
        else:
            ystep = len(y) // 10
        if type(y[0]) in [str, np.str_]:
            yticklabels = y[::ystep]
        else:
            yticklabels = y.round(6)[::ystep]
        plt.yticks(range(0, img.shape[0], ystep), yticklabels)
    plt.title(title)
    plt.show(block=False)

def plot_cm(Y_pred, Y_test, labels, title_prefix='', print_console=True):
    """
    Plot confusion matrix
    """
    N = len(Y_pred)
    cm_norm = 'true' #'true', None
    cm = sklearn.metrics.confusion_matrix(Y_test, Y_pred, normalize=cm_norm, labels=labels)
    cm_raw = sklearn.metrics.confusion_matrix(Y_test, Y_pred, normalize=None, labels=labels)
    img = sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels=labels)
    img.plot(cmap=plt.cm.Blues)
    accuracy = sklearn.metrics.accuracy_score(Y_test, Y_pred)
    f1_score = sklearn.metrics.f1_score(Y_test, Y_pred, average='macro')
    plt.title('%s Confusion matrix (f1=%.2f, acc=%.2f, N=%d)' % (title_prefix, f1_score, accuracy, N))
    if print_console:
        print(sklearn.metrics.classification_report(Y_test, Y_pred, labels=labels))
        print('%s Confusion matrix (raw values)' % title_prefix)
        print(cm_raw)
    plt.show(block=False)
    plt.ioff()
    plt.pause(0.05)

def plot_cm_le(Y_pred, Y_test, le, title_prefix='', print_console=True):
    """
    Plot confusion matrix using label encoder
    """
    N = len(Y_pred)
    cm_norm = 'true' #'true', None
    cm = sklearn.metrics.confusion_matrix(Y_test, Y_pred, normalize=cm_norm)
    img = sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    img.plot(cmap=plt.cm.Blues)
    accuracy = sklearn.metrics.accuracy_score(Y_test, Y_pred)
    f1_score = sklearn.metrics.f1_score(Y_test, Y_pred, average='macro')
    plt.title('%s Confusion matrix (f1=%.2f, acc=%.2f, N=%d)' % (title_prefix, f1_score, accuracy, N))
    if print_console:
        print(sklearn.metrics.classification_report(Y_test, Y_pred, target_names=le.classes_))
        print(sklearn.metrics.confusion_matrix(Y_test, Y_pred, normalize=None))
        print('%s Confusion matrix (raw values)' % title_prefix)
    plt.show(block=False)
    plt.ioff()
    plt.pause(0.05)

def plot_errorbar(data, method='std', ticks=None, title=None, **kwargs):
    """
    Plot horizontal errorbar from an array of samples

    data: Numpy array of dimension [samples] x [variables]
    method:
      sem: standard error of means
      std: standard deviation
    ticks: bar tick labels
    """
    if type(data) == list:
        data = np.array(data)
    elif type(data) != np.ndarray:
        raise ValueError('Unsupported data type %s' % type(data))
    if len(data.shape) != 2:
        raise ValueError('Input must be of 2 dimensions')
    if method == 'std':
        n = 1
    elif method == 'sem':
        n = len(data)
    else:
        raise ValueError('Unsupported method: %s' % method)
    if ticks is None:
        ticks = ['%d'%x for x in range(data.shape[1])]
    if title is None:
        title = ''
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    plt.bar(x=ticks, height=mean, yerr=std/n, **kwargs)
    plt.title(title)

def pvalue2txt(pvalue, max_precision=3):
    if pvalue >= 10**(-max_precision):
        txt = ('p=%%.%df' % max_precision) % pvalue
    elif pvalue == 0:
        txt = 'p<10^-32'
    else:
        txt = 'p<=10^%d' % math.ceil(math.log10(pvalue))
    return txt

'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
 ETC
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''

def int2bits(num, nbits=8):
    """ Convert an integer into bits representation. default=8 bits (0-255) """
    return [int(num) >> x & 1 for x in range(nbits - 1, -1, -1)]

def bits2int(bitlist):
    """ Convert a list of bits into an integer """
    out = 0
    for bit in bitlist:
        out = (out << 1) | bit
    return out
