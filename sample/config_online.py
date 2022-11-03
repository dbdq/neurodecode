import os # only needed in this example to get the current working directory

# decoder path
DECODER_FILE = r'%s/fif/classifier/classifier-64bit.pkl' % os.getcwd()

# use a mock decoder?
## TODO: set DECODER_FILE to None to enable mock decoding
MOCK_CLS = None
DIRECTIONS = [ ('L', 'LEFT_GO'), ('R', 'RIGHT_GO') ]

# trigger device
TRIGGER_DEVICE = 'MOCK' # None | 'ARDUINO' | 'USB2LPT' | 'DESKTOP' | 'SOFTWARE' | 'MOCK'
TRIGGER_FILE = r'%s/mi_left_right_events.ini' % os.getcwd()

# trial settings
TRIALS_EACH = 10 # trials per run
TRIALS_RANDOMIZE = True # randomised trials
TRIALS_RETRY = False
TRIALS_PAUSE = False
SHOW_CUE = True # show the cross cue
SHOW_RESULT = True # show the classification result
SHOW_TRIALS = True # show trial numbers
TIMINGS = { # duration in seconds
    'INIT':2,
    'GAP': 2,
    'READY': 2,
    'FEEDBACK': 1,
    'DIR_CUE': 1,
    'CLASSIFY': 60
}

'''
Feedback type ('BAR' or 'IMAGE')

BAR provides the standard increasing bar feedback.
IMAGE allows the use of custom images as feedback that correspond to each bar position.
For example, to provide an avatar lifting an arm, you can provide 100 images of a person lifting
the right arm and another 100 images of lifting the left arm, and 1 image in neutral position.
Each image correspond to each bar position of 0-100 and pre-loaded images are shown to the user.
Please see /neurodecode/protocols/viz_image.py for details.
'''
FEEDBACK_TYPE = 'BAR' # BAR | IMAGE
IMAGE_PATH = ''


'''
Bar behavior

The class is determined when the bar length reaches 100.

Tips:
1. Free-style mode
It can run in a "free-style" mode until timeout if BAR_REACH_FINISH is False.
This is useful for testing the response of the subject to observe the decoder behaviour.

2. Positive feedback mode and bar biases
This is useful to selectively train the subject in case there's bias.

'''
PROB_ALPHA_NEW = 0.02 # p_smooth = p_old * (1-PROB_ALPHA_NEW) + p_new * PROB_ALPHA_NEW
BAR_BIAS = ('L', 0.0) # BAR_BIAS: None or (dir, prob)

BAR_STEP = {'left':5, 'right':5, 'up':5, 'down':5, 'both':5}
BAR_SLOW_START = {'selected':'False', 'False':None, 'True':[1.0]}            # BAR_SLOW_START: None or in seconds
# finish the trial if bar reaches the end
BAR_REACH_FINISH = False

# positive feedback only?
POSITIVE_FEEDBACK = False

# screen property
SCREEN_SIZE = (1920, 1080)
SCREEN_POS = (0, 0)

# use Google Glass?
GLASS_USE = False

# debug likelihoods
DEBUG_PROBS = True
LOG_PROBS = True

# high frequency parallel decoding (None or dict)
#PARALLEL_DECODING = None
PARALLEL_DECODING = {'selected':'False', 'False':None, 'True':{'period':0.06, 'num_strides':3}}

# visualization refresh rate
REFRESH_RATE = 30 # maximum refresh rate in Hz
