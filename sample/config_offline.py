import os # only needed in this example to get the current working directory

TRIGGER_DEVICE = 'MOCK'
TRIGGER_FILE = r'%s/mi_left_right_events.ini' % os.getcwd()

FEEDBACK_TYPE = 'BAR'
FEEDBACK_IMAGE_PATH = ''

SCREEN_SIZE = (1920, 1080)
SCREEN_POS = (0, 0)
REFRESH_RATE = 50

DIRECTIONS = ['L', 'R']
DIR_RANDOM = False

TRIALS_EACH = 15
TRIAL_PAUSE = False

TIMINGS = { 'INIT':2,           \
            'GAP': 3,           \
            'CUE': 1,            \
            'READY': 1,         \
            'READY_RANDOMIZE':0,    \
            'DIR': 5,           \
            'DIR_RANDOMIZE': 0}   

GLASS_USE = False