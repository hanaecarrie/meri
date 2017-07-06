# coding: utf-8
""" Private module that define usefull wrapper for grid_search function.
"""

try:
    import pisap
    def wrapper_isap(recons_func, **kwargs):
        """ Helper to parallelize the reconstruction.
        """
        res = recons_func(**kwargs)
        if isinstance(res, pisap.base.image.Image):
            return res
        elif isinstance(res, tuple):
            return res[0]
        else:
            raise ValueError(
                "res of 'recons_func' not understood: got {0}".format(type(res)))
except ImportError:
    pass


def _default_wrapper(recons_func, **kwargs):
    """ Default wrapper to parallelize the image reconstruction.
    """
    return recons_func(**kwargs)


def _wrap_im_metrics_func(metrics_funcs, img, ref):
    """ Helper to parallelize the metrics computations.
    """
    if callable(metrics_funcs):
        metrics_funcs = [metrics_funcs]
    err = {}
    for metrics_func in metrics_funcs:
        err[metrics_func.func_name] = metrics_func(img, ref)
    return err
