
"""
Various metrics for evaluating models trained on SEVIR
"""

import numpy as np

"""
Standard contingincy table-based metrics used in forecast verification
https://www.nws.noaa.gov/oh/rfcdev/docs/Glossary_Verification_Metrics.pdf
"""

def probability_of_detection(y_true,y_pred,threshold):
    """

    Inputs:
    -------
    y_true:     [N,L,L,D]
    y_pred:     [N,L,L,D]
    threshold:  [D,]
    Outputs
    -------
    pod=hits/(hits+misses) averaged over the D channels
    """
    return np.mean(run_metric_over_channels(y_true,y_pred,threshold,_pod))

def success_rate(y_true,y_pred,threshold):
    """
    a.k.a    1 - (false alarm rate)
    Inputs:
    -------
    y_true:     [N,L,L,D]
    y_pred:     [N,L,L,D]
    threshold:  [D,]
    Outputs
    -------
    sucr=hits/(hits+false_alarms) averaged over the D channels
    """
    return np.mean(run_metric_over_channels(y_true,y_pred,threshold,_sucr))

def critical_success_index(y_true,y_pred,threshold):
    """

    Inputs:
    -------
    y_true:     [N,L,L,D]
    y_pred:     [N,L,L,D]
    threshold:  [D,]
    Outputs
    -------
    pod=hits/(hits+misses+false_alarms) averaged over the D channels
    """
    return np.mean(run_metric_over_channels(y_true,y_pred,threshold,_csi))

def BIAS(y_true,y_pred,threshold):
    """
    Computes the 2^( mean(log BIAS/log 2) )

    Inputs:
    -------
    y_true:     [N,L,L,D]
    y_pred:     [N,L,L,D]
    threshold:  [D,]
    Outputs
    -------
    pod=(hits+false_alarms)/(hits+misses) pow(2)-log-averaged over the D channels
    """
    logbias = np.log(run_metric_over_channels(y_true,y_pred,threshold,_bias))/np.log(2.0)
    return np.power( 2.0, np.mean(logbias))

def run_metric_over_channels(y_true,y_pred,threshold,metric):
    """

    Inputs:
    -------
    y_true:     [N,L,L,D]
    y_pred:     [N,L,L,D]
    threshold:  [D,]
    Outputs
    -------
    [D,] tensor of metrics computed over each channel
    """
    # Average over channels
    return metric((y_true[:,:,:,0], y_pred[:,:,:,0], threshold))


def _pod(X):
    """
    Single channel version of probability_of_detection
    Inputs:
    -------
    tuple X = (y_true,y_pred,T) where
                        y_true:     [N,L,L]
                        y_pred:     [N,L,L]
                        T:          [1,]
    """
    y_true,y_pred,T=X
    t,p=_threshold(y_true,y_pred,T)
    hits = np.sum(t*p)
    misses = np.sum( t*(1-p) )
    return (hits+1e-6)/(hits+misses+1e-6)


def _sucr(X):
    """
    Single channel version of success_rate
    Inputs:
    -------
    tuple X = (y_true,y_pred,T) where
                        y_true:     [N,L,L]
                        y_pred:     [N,L,L]
                        T:          [1,]
    """
    y_true,y_pred,T=X
    t,p=_threshold(y_true,y_pred,T)
    hits = np.sum(t*p)
    fas = np.sum( (1-t)*p )
    return (hits+1e-6)/(hits+fas+1e-6)

def _csi(X):
    """
    Single channel version of csi
    Inputs:
    -------
    tuple X = (y_true,y_pred,T) where
                        y_true:     [N,L,L]
                        y_pred:     [N,L,L]
                        T:          [1,]
    """
    y_true,y_pred,T=X
    t,p=_threshold(y_true,y_pred,T)
    hits = np.sum(t*p)
    misses = np.sum( t*(1-p) )
    fas = np.sum( (1-t)*p )
    return (hits+1e-6)/(hits+misses+fas+1e-6)

def _bias(X):
    """
    Single channel version of csi
    Inputs:
    -------
    tuple X = (y_true,y_pred,T) where
                        y_true:     [N,L,L]
                        y_pred:     [N,L,L]
                        T:          [1,]
    """
    y_true,y_pred,T=X
    t,p=_threshold(y_true,y_pred,T)
    hits = np.sum(t*p)
    misses = np.sum( t*(1-p) )
    fas = np.sum( (1-t)*p )
    return (hits+fas+1e-6)/(hits+misses+1e-6)

def _threshold(X,Y,T):
    """
    Returns binary tensors t,p the same shape as X & Y.  t = 1 whereever
    X > t.  p =1 wherever Y > t.  p and t are set to 0 whereever EITHER
    t or p are nan.   This is useful for counts that don't involve correct
    rejections.
    """
    t=np.greater_equal(X, T)
    #t=np.dtypes.cast(t, np.float32)
    p=np.greater_equal(Y, T)
    #p=np.dtypes.cast(p, np.float32)
    is_nan = np.logical_or(np.isnan(X),np.isnan(Y))
    t = np.where(is_nan,np.zeros_like(t),t)
    p = np.where(is_nan,np.zeros_like(p),p)
    return t,p
