import numpy as np

# Import from neurips-2020-sevir repository
from metrics.util import probability_of_detection, success_rate, critical_success_index, BIAS


def get_metric_functions():
    from functools import partial

    def pod74(yt, yp):
        return probability_of_detection( yt, yp, np.array([74.0]))

    def pod133(yt, yp):
        return probability_of_detection(yt, yp, np.array([133.0]))

    def sucr74(yt, yp):
        return success_rate(yt, yp, np.array([74.0]))

    def sucr133(yt, yp):
        return success_rate(yt, yp, np.array([133.0]))

    def csi74(yt, yp):
        return critical_success_index(yt, yp, np.array([74.0]))

    def csi133(yt, yp):
        return critical_success_index(yt, yp, np.array([133.0]))

    def bias74(yt, yp):
        return BIAS(yt, yp, np.array([74.0]))

    def bias133(yt, yp):
        return BIAS(yt, yp, np.array([133.0]))

    metric_functions = {}
    metric_functions['pod74'] = pod74
    metric_functions['pod133'] = pod133
    metric_functions['sucr74'] = sucr74
    metric_functions['sucr133'] = sucr133
    metric_functions['csi74'] = csi74
    metric_functions['csi133'] = csi133
    metric_functions['bias74'] = bias74
    metric_functions['bias133'] = bias133
    return metric_functions


def add_metric_vals(metrics_dict, keys, vals):
    for key, val in zip(keys, vals):
        metrics_dict[key].append(val)


def compute_metrics(y_true, y_pred, metric_functions):
    y_true_t, y_pred_t = y_true.permute(0, 2, 3, 1).cpu().detach().numpy(), y_pred.permute(0, 2, 3, 1).cpu().detach().numpy()

    metric_vals = {}
    for key, func in metric_functions.items():
        metric_vals[key] = func(y_true_t, y_pred_t)

    return metric_vals


def compute_avg_metric_vals(metrics_dict):
    metrics_dict_avg = {}
    for key in metrics_dict:
        metrics_dict_avg[key] = np.mean(metrics_dict[key])
    return metrics_dict_avg


def print_avg_metric_vals(metrics_dict, mode='train'):
    avg_vals = compute_avg_metric_vals(metrics_dict)
    for key in avg_vals:
        print('{}_{}: {}'.format(
            key, mode, avg_vals[key]))
    print()

