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

Author: Kyuhwa Lee
"""

import numpy as np

class ScalerMad:
    def __init__(self):
        self.median = 0
        self.mad_corrected = 1

    def fit(self, X, axis=None):
        """
        Compute and save the statistics along the specified axis

        input
        -----
        X: numpy array
        axis: axis to be applied
              if None, compute along all axes, i.e. flattened version of the array
        """
        # compute median
        self.median = np.median(X, axis=axis, keepdims=True)
        # compute MAD
        mad = np.median(np.abs(X - self.median), axis=axis, keepdims=True)

        if mad.all():
            # if all MADs are non-zero
            self.mad_corrected = 1.486 * mad
        else:
            # compute MeanAD
            X_mean = np.mean(X, axis=axis, keepdims=True)
            X_mean_adj = np.mean(abs(X - X_mean), axis=axis, keepdims=True)
            self.mad_corrected = 1.486 * mad
            # replace 0 in MAD with the corresponding meanAD value
            for zero_loc in np.array(np.where(mad==0)).T:
                zero_loc = tuple(zero_loc) # each index containing 0 in MAD
                self.mad_corrected[zero_loc] = 1.253314 * X_mean_adj[zero_loc]

    def fit_transform(self, X, axis=None):
        """
        Perform fit() and transform() at once
        """
        self.fit(X, axis)
        return self.transform(X)

    def transform(self, X):
        """
        Normalise using median and MAD
        """
        return (X - self.median) / self.mad_corrected

    def inverse_transform(self, X_norm):
        """
        Inverse normalisation
        """
        return X_norm * self.mad_corrected + self.median

    def get_median(self):
        """
        Return the computed median
        """
        return self.median

    def get_mad(self):
        """
        Return the computed MAD
        """
        return self.mad_corrected
