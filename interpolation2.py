# Modified version of the interpolation package
# The linear interpolation is returning the weights on top of the interpolated field.


from __future__ import division, print_function, absolute_import

__all__ = ['interp1d', 'interp2d', 'lagrange', 'PPoly', 'BPoly', 'NdPPoly',
           'RegularGridInterpolator', 'interpn']

import itertools
import warnings
import functools
import operator

import numpy as np
from numpy import (array, transpose, searchsorted, atleast_1d, atleast_2d,
                   ravel, poly1d, asarray, intp)

import scipy.special as spec
from scipy.special import comb

from scipy._lib.six import xrange, integer_types, string_types

#from . import fitpack
#from . import dfitpack
#from . import _fitpack
#from .polyint import _Interpolator1D
#from . import _ppoly
#from .fitpack2 import RectBivariateSpline
#from .interpnd import _ndim_coords_from_arrays
#from ._bsplines import make_interp_spline, BSpline





class RegularGridInterpolator(object):
    """
    Interpolation on a regular grid in arbitrary dimensions

    The data must be defined on a regular grid; the grid spacing however may be
    uneven.  Linear and nearest-neighbour interpolation are supported. After
    setting up the interpolator object, the interpolation method (*linear* or
    *nearest*) may be chosen at each evaluation.

    Parameters
    ----------
    points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
        The points defining the regular grid in n dimensions.

    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions.

    method : str, optional
        The method of interpolation to perform. Supported are "linear" and
        "nearest". This parameter will become the default for the object's
        ``__call__`` method. Default is "linear".

    bounds_error : bool, optional
        If True, when interpolated values are requested outside of the
        domain of the input data, a ValueError is raised.
        If False, then `fill_value` is used.

    fill_value : number, optional
        If provided, the value to use for points outside of the
        interpolation domain. If None, values outside
        the domain are extrapolated.

    Methods
    -------
    __call__

    Notes
    -----
    Contrary to LinearNDInterpolator and NearestNDInterpolator, this class
    avoids expensive triangulation of the input data by taking advantage of the
    regular grid structure.

    If any of `points` have a dimension of size 1, linear interpolation will
    return an array of `nan` values. Nearest-neighbor interpolation will work
    as usual in this case.

    .. versionadded:: 0.14

    Examples
    --------
    Evaluate a simple example function on the points of a 3D grid:

    >>> from scipy.interpolate import RegularGridInterpolator
    >>> def f(x, y, z):
    ...     return 2 * x**3 + 3 * y**2 - z
    >>> x = np.linspace(1, 4, 11)
    >>> y = np.linspace(4, 7, 22)
    >>> z = np.linspace(7, 9, 33)
    >>> data = f(*np.meshgrid(x, y, z, indexing='ij', sparse=True))

    ``data`` is now a 3D array with ``data[i,j,k] = f(x[i], y[j], z[k])``.
    Next, define an interpolating function from this data:

    >>> my_interpolating_function = RegularGridInterpolator((x, y, z), data)

    Evaluate the interpolating function at the two points
    ``(x,y,z) = (2.1, 6.2, 8.3)`` and ``(3.3, 5.2, 7.1)``:

    >>> pts = np.array([[2.1, 6.2, 8.3], [3.3, 5.2, 7.1]])
    >>> my_interpolating_function(pts)
    array([ 125.80469388,  146.30069388])

    which is indeed a close approximation to
    ``[f(2.1, 6.2, 8.3), f(3.3, 5.2, 7.1)]``.

    See also
    --------
    NearestNDInterpolator : Nearest neighbour interpolation on unstructured
                            data in N dimensions

    LinearNDInterpolator : Piecewise linear interpolant on unstructured data
                           in N dimensions

    References
    ----------
    .. [1] Python package *regulargrid* by Johannes Buchner, see
           https://pypi.python.org/pypi/regulargrid/
    .. [2] Wikipedia, "Trilinear interpolation",
           https://en.wikipedia.org/wiki/Trilinear_interpolation
    .. [3] Weiser, Alan, and Sergio E. Zarantonello. "A note on piecewise linear
           and multilinear table interpolation in many dimensions." MATH.
           COMPUT. 50.181 (1988): 189-196.
           https://www.ams.org/journals/mcom/1988-50-181/S0025-5718-1988-0917826-0/S0025-5718-1988-0917826-0.pdf

    """
    # this class is based on code originally programmed by Johannes Buchner,
    # see https://github.com/JohannesBuchner/regulargrid

    def __init__(self, points, values, method="linear", bounds_error=True,
                 fill_value=np.nan):
        if method not in ["linear", "nearest"]:
            raise ValueError("Method '%s' is not defined" % method)
        self.method = method
        self.bounds_error = bounds_error

        if not hasattr(values, 'ndim'):
            # allow reasonable duck-typed values
            values = np.asarray(values)

        if len(points) > values.ndim:
            raise ValueError("There are %d point arrays, but values has %d "
                             "dimensions" % (len(points), values.ndim))

        if hasattr(values, 'dtype') and hasattr(values, 'astype'):
            if not np.issubdtype(values.dtype, np.inexact):
                values = values.astype(float)

        self.fill_value = fill_value
        if fill_value is not None:
            fill_value_dtype = np.asarray(fill_value).dtype
            if (hasattr(values, 'dtype') and not
                    np.can_cast(fill_value_dtype, values.dtype,
                                casting='same_kind')):
                raise ValueError("fill_value must be either 'None' or "
                                 "of a type compatible with values")

        for i, p in enumerate(points):
            if not np.all(np.diff(p) > 0.):
                raise ValueError("The points in dimension %d must be strictly "
                                 "ascending" % i)
            if not np.asarray(p).ndim == 1:
                raise ValueError("The points in dimension %d must be "
                                 "1-dimensional" % i)
            if not values.shape[i] == len(p):
                raise ValueError("There are %d points and %d values in "
                                 "dimension %d" % (len(p), values.shape[i], i))
        self.grid = tuple([np.asarray(p) for p in points])
        self.values = values

    def __call__(self, xi, method=None):
        """
        Interpolation at coordinates

        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at

        method : str
            The method of interpolation to perform. Supported are "linear" and
            "nearest".

        """
        method = self.method if method is None else method
        if method not in ["linear", "nearest"]:
            raise ValueError("Method '%s' is not defined" % method)

        ndim = len(self.grid)
        #xi = _ndim_coords_from_arrays(xi, ndim=ndim)
        if xi.shape[-1] != len(self.grid):
            raise ValueError("The requested sample points xi have dimension "
                             "%d, but this RegularGridInterpolator has "
                             "dimension %d" % (xi.shape[1], ndim))

        xi_shape = xi.shape
        xi = xi.reshape(-1, xi_shape[-1])

        if self.bounds_error:
            for i, p in enumerate(xi.T):
                if not np.logical_and(np.all(self.grid[i][0] <= p),
                                      np.all(p <= self.grid[i][-1])):
                    raise ValueError("One of the requested xi is out of bounds "
                                     "in dimension %d" % i)

        indices, norm_distances, out_of_bounds = self._find_indices(xi.T)
        if method == "linear":
            result, der = self._evaluate_linear(indices,
                                           norm_distances,
                                           out_of_bounds)
        else:
            raise ValueError("'linear' is the only method supported")

        return result.reshape(xi_shape[:-1] + self.values.shape[ndim:]), der.reshape(xi_shape[:-1] + self.values.shape[ndim:])

    def _evaluate_linear(self, indices, norm_distances, out_of_bounds):
        # slice for broadcasting over trailing dimensions in self.values
        vslice = (slice(None),) + (None,)*(self.values.ndim - len(indices))

        # find relevant values
        # each i and i+1 represents a edge
        edges = itertools.product(*[[i, i + 1] for i in indices])
        values = 0.
        values_t = 0.

        for edge_indices in edges:
            #print(' 123   ')
            #print(edges)
            #print(edge_indices)
            weight = 1.
            weight_t = 1.
            for ei, i, yi in zip(edge_indices, indices, norm_distances):
                #print(i.shape)
                #print(yi)
                weight *= np.where(ei == i, 1 - yi, yi)
                weight_t *= np.where(ei == i, -1, 1)
            #print(weight.shape)
            #print(edge_indices)
            #exit()
            values += np.asarray(self.values[edge_indices]) * weight[vslice]
            values_t += np.asarray(self.values[edge_indices]) * weight_t[vslice]
            #print(weight)
            #print(values.shape)
        #exit()
        return values, values_t



    def _find_indices(self, xi):
        # find relevant edges between which xi are situated
        indices = []
        # compute distance to lower edge in unity units
        norm_distances = []
        # check for out of bounds xi
        out_of_bounds = np.zeros((xi.shape[1]), dtype=bool)
        # iterate through dimensions
        for x, grid in zip(xi, self.grid):
            i = np.searchsorted(grid, x) - 1
            i[i < 0] = 0
            i[i > grid.size - 2] = grid.size - 2
            indices.append(i)
            norm_distances.append((x - grid[i]) /
                                  (grid[i + 1] - grid[i]))
            if not self.bounds_error:
                out_of_bounds += x < grid[0]
                out_of_bounds += x > grid[-1]
        return indices, norm_distances, out_of_bounds


def interpn(points, values, xi, method="linear", bounds_error=True,
            fill_value=np.nan):
    """
    Multidimensional interpolation on regular grids.

    Parameters
    ----------
    points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
        The points defining the regular grid in n dimensions.

    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions.

    xi : ndarray of shape (..., ndim)
        The coordinates to sample the gridded data at

    method : str, optional
        The method of interpolation to perform. Supported are "linear" and
        "nearest", and "splinef2d". "splinef2d" is only supported for
        2-dimensional data.

    bounds_error : bool, optional
        If True, when interpolated values are requested outside of the
        domain of the input data, a ValueError is raised.
        If False, then `fill_value` is used.

    fill_value : number, optional
        If provided, the value to use for points outside of the
        interpolation domain. If None, values outside
        the domain are extrapolated.  Extrapolation is not supported by method
        "splinef2d".

    Returns
    -------
    values_x : ndarray, shape xi.shape[:-1] + values.shape[ndim:]
        Interpolated values at input coordinates.

    Notes
    -----

    .. versionadded:: 0.14

    See also
    --------
    NearestNDInterpolator : Nearest neighbour interpolation on unstructured
                            data in N dimensions

    LinearNDInterpolator : Piecewise linear interpolant on unstructured data
                           in N dimensions

    RegularGridInterpolator : Linear and nearest-neighbor Interpolation on a
                              regular grid in arbitrary dimensions

    RectBivariateSpline : Bivariate spline approximation over a rectangular mesh

    """
    # sanity check 'method' kwarg
    if method not in ["linear", "nearest", "splinef2d"]:
        raise ValueError("interpn only understands the methods 'linear', "
                         "'nearest', and 'splinef2d'. You provided %s." %
                         method)

    if not hasattr(values, 'ndim'):
        values = np.asarray(values)

    ndim = values.ndim
    if ndim > 2 and method == "splinef2d":
        raise ValueError("The method spline2fd can only be used for "
                         "2-dimensional input data")
    if not bounds_error and fill_value is None and method == "splinef2d":
        raise ValueError("The method spline2fd does not support extrapolation.")

    # sanity check consistency of input dimensions
    if len(points) > ndim:
        raise ValueError("There are %d point arrays, but values has %d "
                         "dimensions" % (len(points), ndim))
    if len(points) != ndim and method == 'splinef2d':
        raise ValueError("The method spline2fd can only be used for "
                         "scalar data with one point per coordinate")

    # sanity check input grid
    for i, p in enumerate(points):
        if not np.all(np.diff(p) > 0.):
            raise ValueError("The points in dimension %d must be strictly "
                             "ascending" % i)
        if not np.asarray(p).ndim == 1:
            raise ValueError("The points in dimension %d must be "
                             "1-dimensional" % i)
        if not values.shape[i] == len(p):
            raise ValueError("There are %d points and %d values in "
                             "dimension %d" % (len(p), values.shape[i], i))
    grid = tuple([np.asarray(p) for p in points])

    # sanity check requested xi
    xi = _ndim_coords_from_arrays(xi, ndim=len(grid))
    if xi.shape[-1] != len(grid):
        raise ValueError("The requested sample points xi have dimension "
                         "%d, but this RegularGridInterpolator has "
                         "dimension %d" % (xi.shape[1], len(grid)))

    for i, p in enumerate(xi.T):
        if bounds_error and not np.logical_and(np.all(grid[i][0] <= p),
                                               np.all(p <= grid[i][-1])):
            raise ValueError("One of the requested xi is out of bounds "
                             "in dimension %d" % i)

    # perform interpolation
    if method == "linear":
        interp = RegularGridInterpolator(points, values, method="linear",
                                         bounds_error=bounds_error,
                                         fill_value=fill_value)
        return interp(xi)
    elif method == "nearest":
        interp = RegularGridInterpolator(points, values, method="nearest",
                                         bounds_error=bounds_error,
                                         fill_value=fill_value)
        return interp(xi)
    elif method == "splinef2d":
        xi_shape = xi.shape
        xi = xi.reshape(-1, xi.shape[-1])

        # RectBivariateSpline doesn't support fill_value; we need to wrap here
        idx_valid = np.all((grid[0][0] <= xi[:, 0], xi[:, 0] <= grid[0][-1],
                            grid[1][0] <= xi[:, 1], xi[:, 1] <= grid[1][-1]),
                           axis=0)
        result = np.empty_like(xi[:, 0])

        # make a copy of values for RectBivariateSpline
        interp = RectBivariateSpline(points[0], points[1], values[:])
        result[idx_valid] = interp.ev(xi[idx_valid, 0], xi[idx_valid, 1])
        result[np.logical_not(idx_valid)] = fill_value

        return result.reshape(xi_shape[:-1])

#########################################################################################################
#========================================================================================================
def interpn_linear(points, values, xi, method="linear", bounds_error=True,
            fill_value=np.nan):
    """
    Multidimensional interpolation on regular grids.

    Parameters
    ----------
    points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
        The points defining the regular grid in n dimensions.

    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions.

    xi : ndarray of shape (..., ndim)
        The coordinates to sample the gridded data at

    method : str, optional
        The method of interpolation to perform. Supported are "linear" and
        "nearest", and "splinef2d". "splinef2d" is only supported for
        2-dimensional data.

    bounds_error : bool, optional
        If True, when interpolated values are requested outside of the
        domain of the input data, a ValueError is raised.
        If False, then `fill_value` is used.

    fill_value : number, optional
        If provided, the value to use for points outside of the
        interpolation domain. If None, values outside
        the domain are extrapolated.  Extrapolation is not supported by method
        "splinef2d".

    Returns
    -------
    values_x : ndarray, shape xi.shape[:-1] + values.shape[ndim:]
        Interpolated values at input coordinates.


    """
    # sanity check 'method' kwarg
    if method not in ["linear"]:
        raise ValueError("interpn only understands the methods 'linear'. You provided %s." %
                         method)

    if not hasattr(values, 'ndim'):
        values = np.asarray(values)

    ndim = values.ndim

    # sanity check consistency of input dimensions
    if len(points) > ndim:
        raise ValueError("There are %d point arrays, but values has %d "
                         "dimensions" % (len(points), ndim))
    if len(points) != ndim and method == 'splinef2d':
        raise ValueError("The method spline2fd can only be used for "
                         "scalar data with one point per coordinate")

    # sanity check input grid
    for i, p in enumerate(points):
        if not np.all(np.diff(p) > 0.):
            raise ValueError("The points in dimension %d must be strictly "
                             "ascending" % i)
        if not np.asarray(p).ndim == 1:
            raise ValueError("The points in dimension %d must be "
                             "1-dimensional" % i)
        if not values.shape[i] == len(p):
            raise ValueError("There are %d points and %d values in "
                             "dimension %d" % (len(p), values.shape[i], i))
    grid = tuple([np.asarray(p) for p in points])

    # sanity check requested xi
    #xi = _ndim_coords_from_arrays(xi, ndim=len(grid))
    #if xi.shape[-1] != len(grid):
    #    raise ValueError("The requested sample points xi have dimension "
    #                     "%d, but this RegularGridInterpolator has "
    #                     "dimension %d" % (xi.shape[1], len(grid)))

    for i, p in enumerate(xi.T):
        if bounds_error and not np.logical_and(np.all(grid[i][0] <= p),
                                               np.all(p <= grid[i][-1])):
            raise ValueError("One of the requested xi is out of bounds "
                             "in dimension %d" % i)

    # perform interpolation
    interp = RegularGridInterpolator(points, values, method="linear",
                                         bounds_error=bounds_error,
                                         fill_value=fill_value)
    return interp(xi)

