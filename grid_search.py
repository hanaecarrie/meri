# coding: utf-8
""" Module that declare the gridding search parameters function.
"""
# system import
import sys
import itertools
import psutil
import humanize
import numpy as np
from joblib import Parallel, delayed

# package import
from ._wrap import _wrap_im_metrics_func, _default_wrapper
from .reporting import ReportGridSearch


def _get_final_size(param_grid):
    """ Return the memory size of the given param_grid when it will extend as
    a carthesian grid a parameters.

    Parameters:
    ----------
    param_grid: dict or list of dictionaries,
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values.

    Return:
    -------
    size: int,
        the number of bytes of the extended carthesian grid a parameters.
    """
    tmp = {} # same pattern than param_grid but store the size
    for idx, key in enumerate(param_grid.iterkeys()):
        if isinstance(param_grid[key], list):
            tmp[idx] = [sys.getsizeof(value) for value in param_grid[key]]
        else:
            tmp[idx] = [sys.getsizeof(param_grid[key])]
    return np.array([x for x in itertools.product(*tmp.values())]).sum()


def grid_search(recons_func, param_grid, metrics_funcs, metrics_direction,
                im_ref, wrapper=None, n_jobs=1, verbose=0):
    """ Run `recons_func` on the carthesian product of `param_grid` then run
        each error measurement function in `metrics_funcs` w.r.t `im_ref`.

        Notes:
        -----
        `recons_func` return should be handle by wrapper if it's not a simple
        np.ndarray image.

        Parameters:
        -----------
        recons_func: function,
            The reconstruction function from whom to tune the hyperparameters.
            `recons_func` return should be handle by wrapper if it's not a
            simple np.ndarray image.
        param_grid: dict or list of dictionaries,
            Dictionary with parameters names (string) as keys and lists of
            parameter settings to try as values: the grids spanned by each
            dictionary in the list are explored.
        metrics_funcs: list of functions,
            The list of functions for the error measurement. Each one should
            only accept two arguments: im and ref and each function should
            return a real number.
        metrics_direction: list of bool,
            Specify if the metrics mean: True if the lower the better, False
            for the greater the better. It will be directly pass to the report
            result.
        im_ref: np.ndarray
            This image reference for computing the error for each functions in
            metrics_funcs.
        wrapper: func, (default: None)
            Handle the call of recons_func if some pre-process or post-process
            should be done. `wrapper` has a specific API:
            `wrapper(recons_func, **kwargs)`
        n_jobs: int (default: 1),
            The maximum number of concurrently running jobs, such as the number
            of Python worker processes when backend=”multiprocessing” or the
            size of the thread-pool when backend=”threading”. If -1 all CPUs
            are used. If 1 is given, no parallel computing code is used at all,
            which is useful for debugging. For n_jobs below -1,
            (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2,
            all CPUs but one are used.
        verbose: int (default: 0),
            The verbosity level: if non zero, progress messages are printed.
            Above 50, the output is sent to stdout. The frequency of the
            messages increases with the verbosity level. If it more than 10,
            all iterations are reported.

        Results:
        --------
        report: class,
            A reporting class containing all the reconstructed image, the given
            parameters, the error measurement, and differents methods to select
            the best parameters w.r.t a specific metrics.
    """
    if wrapper is None:
        wrapper = _default_wrapper
    # check if enough memory
    size_ = _get_final_size(param_grid)
    if size_ > 0.9 * psutil.virtual_memory().available:
        raise MemoryError("not enough memory 'param_grid'"
                          " weigh {0} ..".format(humanize.naturalsize(size_)))
    # sanitize value to list type
    for key, value in param_grid.iteritems():
        if not isinstance(value, list):
            param_grid[key] = [value]
    list_kwargs = [dict(zip(param_grid, x))
                   for x in itertools.product(*param_grid.values())]
    if verbose > 0:
        print(("Running {0} metrics for {1} candidates, totalling {1} "
               "reconstruction and {2} metrics"
               " computations").format(len(metrics_funcs),
                                       len(list_kwargs),
                                       len(list_kwargs) * len(metrics_funcs),
                                      ))
    # Run the reconstruction
    recons_im = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(wrapper)(recons_func, **kwargs)
                for kwargs in list_kwargs)
    # Compute the metrics
    errs = Parallel(n_jobs=n_jobs, verbose=verbose)(
           delayed(_wrap_im_metrics_func)(metrics_funcs, im.data, im_ref)
           for im in recons_im)
    metrics_direction = dict(zip([func.func_name for func in metrics_funcs],
                                 metrics_direction))
    return ReportGridSearch(list_kwargs, param_grid, recons_im, im_ref,
                            metrics_funcs, metrics_direction, errs)
