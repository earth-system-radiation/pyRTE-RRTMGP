import copy

import numpy as np
import pytest

import pyrte_rrtmgp.pyrte_rrtmgp as py

#####################
## test_zero_array ##
#####################


@pytest.mark.parametrize(
    "array,method,error",
    [
        (np.ones((3, 3)), py.zero_array_1D, "Number of dimensions must be one"),
        (np.ones((3, 3, 3)), py.zero_array_2D, "Number of dimensions must be 2"),
        (np.ones((3, 3)), py.zero_array_3D, "Number of dimensions must be 3"),
        (np.ones((3, 3)), py.zero_array_4D, "Number of dimensions must be 4"),
    ],
)
def test_invalid_array_dimension(array, method, error):
    with pytest.raises(RuntimeError) as excinfo:
        method(array)
    assert str(excinfo.value) == error


@pytest.mark.parametrize(
    "array,method",
    [
        (np.ones((0,)), py.zero_array_1D),
        (np.ones((0, 0)), py.zero_array_2D),
        (np.ones((0, 0, 0)), py.zero_array_3D),
        (np.ones((0, 0, 0, 0)), py.zero_array_4D),
    ],
)
def test_empty_array_exception(array, method):
    with pytest.raises(RuntimeError) as excinfo:
        method(array)
    assert str(excinfo.value) == "Array size cannot be 0 or negative"


@pytest.mark.parametrize(
    "shape, fortran_zero_array",
    [
        ((4,), py.zero_array_1D),
        ((4, 4), py.zero_array_2D),
        ((4, 4, 4), py.zero_array_3D),
        ((4, 4, 4, 4), py.zero_array_4D),
    ],
)
def test_zero_array(shape, fortran_zero_array):
    arr = np.random.rand(*shape)
    fortran_zero_array(arr)
    assert np.all(arr == 0)


###########################################
## test_rte_increment_1scalar_by_1scalar ##
###########################################


def test_rte_increment_1scalar_by_1scalar_dimension_check():

    ncol = 0
    nlay = 0
    ngpt = 0
    tau_inout = np.random.rand(ncol, nlay, ngpt)
    tau_in = np.random.rand(ncol, nlay, ngpt)

    with pytest.raises(RuntimeError) as excinfo:
        py.rte_increment_1scalar_by_1scalar(ncol, nlay, ngpt, tau_inout, tau_in)
    assert str(excinfo.value) == "ncol, nlay, and ngpt must be positive integers"


@pytest.mark.parametrize(
    "parameters,method",
    [
        (
            (3, 3, 3, np.ones((3, 3)), np.ones((3, 3))),
            py.rte_increment_1scalar_by_1scalar,
        )
    ],
)
def test_rte_increment_1scalar_by_1scalar_array_size_check(parameters, method):
    with pytest.raises(RuntimeError) as excinfo:
        method(*parameters)
    assert str(excinfo.value) == "Invalid size for input arrays"


def test_rte_increment_1scalar_by_1scalar():
    ncol, nlay, ngpt = (3, 4, 5)
    tau_inout = np.random.rand(ncol, nlay, ngpt)
    tau_in = np.random.rand(ncol, nlay, ngpt)

    parameters = [ncol, nlay, ngpt, tau_inout, tau_in]

    res = np.array(tau_inout)

    for igpt in range(ngpt):
        for ilay in range(nlay):
            for icol in range(ncol):
                res[icol][ilay][igpt] += tau_in[icol][ilay][igpt]

    py.rte_increment_1scalar_by_1scalar(*parameters)
    assert np.array_equal(tau_inout, res)


###########################################
## test_rte_increment_1scalar_by_2stream ##
###########################################


def test_rte_increment_1scalar_by_2stream_dimension_check():
    ncol = 0
    nlay = 0
    ngpt = 0
    tau_inout = np.random.rand(ncol, nlay, ngpt)
    tau_in = np.random.rand(ncol, nlay, ngpt)
    ssa_in = np.random.rand(ncol, nlay, ngpt)
    with pytest.raises(RuntimeError) as excinfo:
        py.rte_increment_1scalar_by_2stream(ncol, nlay, ngpt, tau_inout, tau_in, ssa_in)
    assert str(excinfo.value) == "ncol, nlay, and ngpt must be positive integers"


@pytest.mark.parametrize(
    "parameters,method",
    [
        (
            (3, 3, 3, np.ones((3, 3)), np.ones((3, 3)), np.ones((3, 3))),
            py.rte_increment_1scalar_by_2stream,
        )
    ],
)
def test_rte_increment_1scalar_by_2stream_array_size_check(parameters, method):
    with pytest.raises(RuntimeError) as excinfo:
        method(*parameters)
    assert str(excinfo.value) == "Invalid size for input arrays"


@pytest.mark.parametrize(
    "parameters, method",
    [
        (
            (3, 4, 5, np.ones((3, 4, 5)), np.ones((3, 4, 5)), np.ones((3, 4, 5))),
            py.rte_increment_1scalar_by_2stream,
        )
    ],
)
def test_rte_increment_1scalar_by_2stream(parameters, method):
    ncol, nlay, ngpt, tau_inout, tau_in, ssa_in = parameters
    res = np.array(tau_inout)

    for igpt in range(ngpt):
        for ilay in range(nlay):
            for icol in range(ncol):
                res[icol, ilay, igpt] += tau_in[icol, ilay, igpt] * (
                    1.0 - ssa_in[icol, ilay, igpt]
                )

    method(*parameters)
    assert np.array_equal(tau_inout, res)


###########################################
## test_rte_increment_1scalar_by_nstream ##
###########################################


def test_rte_increment_1scalar_by_nstream_dimension_check():
    ncol = 0
    nlay = 0
    ngpt = 0
    tau_inout = np.random.rand(ncol, nlay, ngpt)
    tau_in = np.random.rand(ncol, nlay, ngpt)
    ssa_in = np.random.rand(ncol, nlay, ngpt)

    parameters = [ncol, nlay, ngpt, tau_inout, tau_in, ssa_in]

    with pytest.raises(RuntimeError) as excinfo:
        py.rte_increment_1scalar_by_nstream(ncol, nlay, ngpt, tau_inout, tau_in, ssa_in)
    assert str(excinfo.value) == "ncol, nlay, and ngpt must be positive integers"


@pytest.mark.parametrize(
    "parameters,method",
    [
        (
            (3, 3, 3, np.ones((3, 3)), np.ones((3, 3)), np.ones((3, 3))),
            py.rte_increment_1scalar_by_nstream,
        )
    ],
)
def test_rte_increment_1scalar_by_nstream_array_size_check(parameters, method):
    with pytest.raises(RuntimeError) as excinfo:
        method(*parameters)
    assert str(excinfo.value) == "Invalid size for input arrays"


def test_rte_increment_1scalar_by_nstream():
    ncol = 3
    nlay = 4
    ngpt = 5
    tau_inout = np.random.rand(ncol, nlay, ngpt)
    ssa_in = np.random.rand(ncol, nlay, ngpt)
    tau_in = np.random.rand(ncol, nlay, ngpt)

    parameters = [ncol, nlay, ngpt, tau_inout, ssa_in, tau_in]
    res = np.array(tau_inout)

    for igpt in range(ngpt):
        for ilay in range(nlay):
            for icol in range(ncol):
                res[icol - 1, ilay - 1, igpt - 1] += ssa_in[
                    icol - 1, ilay - 1, igpt - 1
                ] * (1.0 - tau_in[icol - 1, ilay - 1, igpt - 1])

    py.rte_increment_1scalar_by_nstream(*parameters)
    assert np.array_equal(tau_inout, res)


###########################################
## test_rte_increment_2stream_by_1scalar ##
###########################################


@pytest.mark.parametrize(
    "parameters,method",
    [
        (
            (0, 0, 0, np.ones((3, 3, 3)), np.ones((3, 3, 3)), np.ones((3, 3, 3))),
            py.rte_increment_2stream_by_1scalar,
        )
    ],
)
def test_rte_increment_2stream_by_1scalar_dimension_check(parameters, method):
    with pytest.raises(RuntimeError) as excinfo:
        method(*parameters)
    assert str(excinfo.value) == "ncol, nlay, and ngpt must be positive integers"


@pytest.mark.parametrize(
    "parameters,method",
    [
        (
            (3, 3, 3, np.ones((3, 3)), np.ones((3, 3)), np.ones((3, 3))),
            py.rte_increment_2stream_by_1scalar,
        )
    ],
)
def test_rte_increment_2stream_by_1scalar_array_size_check(parameters, method):
    with pytest.raises(RuntimeError) as excinfo:
        method(*parameters)
    assert str(excinfo.value) == "Invalid size for input arrays"


def test_rte_increment_2stream_by_1scalar():

    eps = 3.0 * np.finfo(float).tiny

    ncol = 3
    nlay = 4
    ngpt = 5
    tau_inout = np.random.rand(ncol, nlay, ngpt)
    ssa_inout = np.random.rand(ncol, nlay, ngpt)
    tau_in = np.random.rand(ncol, nlay, ngpt)

    parameters = [ncol, nlay, ngpt, tau_inout, ssa_inout, tau_in]

    res_tau_inout = np.array(tau_inout)
    res_ssa_inout = np.array(ssa_inout)

    eps = 3.0 * np.finfo(float).tiny

    for igpt in range(1, ngpt + 1):
        for ilay in range(1, nlay + 1):
            for icol in range(1, ncol + 1):
                tau12 = (
                    tau_inout[icol - 1, ilay - 1, igpt - 1]
                    + tau_in[icol - 1, ilay - 1, igpt - 1]
                )
                res_ssa_inout[icol - 1, ilay - 1, igpt - 1] = (
                    tau_inout[icol - 1, ilay - 1, igpt - 1]
                    * ssa_inout[icol - 1, ilay - 1, igpt - 1]
                    / max(eps, tau12)
                )
                res_tau_inout[icol - 1, ilay - 1, igpt - 1] = tau12

    py.rte_increment_2stream_by_1scalar(*parameters)
    assert np.array_equal(res_tau_inout, tau_inout) and np.array_equal(
        res_ssa_inout, ssa_inout
    )


###########################################
## test_rte_increment_2stream_by_2stream ##
###########################################


@pytest.mark.parametrize(
    "parameters,method",
    [
        (
            (
                0,
                0,
                0,
                np.ones((3, 3, 3)),
                np.ones((3, 3, 3)),
                np.ones((3, 3, 3)),
                np.ones((3, 3, 3)),
                np.ones((3, 3, 3)),
                np.ones((3, 3, 3)),
            ),
            py.rte_increment_2stream_by_2stream,
        )
    ],
)
def test_rte_increment_2stream_by_2stream_dimension_check(parameters, method):
    with pytest.raises(RuntimeError) as excinfo:
        method(*parameters)
    assert str(excinfo.value) == "ncol, nlay, and ngpt must be positive integers"


@pytest.mark.parametrize(
    "parameters,method",
    [
        (
            (
                3,
                3,
                3,
                np.ones((3, 3)),
                np.ones((3, 3)),
                np.ones((3, 3)),
                np.ones((3, 3)),
                np.ones((3, 3)),
                np.ones((3, 3)),
            ),
            py.rte_increment_2stream_by_2stream,
        )
    ],
)
def test_rte_increment_2stream_by_2stream_array_size_check(parameters, method):
    with pytest.raises(RuntimeError) as excinfo:
        method(*parameters)
    assert str(excinfo.value) == "Invalid size for input arrays"


def test_rte_increment_2stream_by_2stream():
    ncol = 3
    nlay = 4
    ngpt = 5
    tau_inout = np.random.rand(ncol, nlay, ngpt)
    ssa_inout = np.random.rand(ncol, nlay, ngpt)
    g_inout = np.random.rand(ncol, nlay, ngpt)
    tau_in = np.random.rand(ncol, nlay, ngpt)
    ssa_in = np.random.rand(ncol, nlay, ngpt)
    g_in = np.random.rand(ncol, nlay, ngpt)

    parameters = [ncol, nlay, ngpt, tau_inout, ssa_inout, g_inout, tau_in, ssa_in, g_in]

    res_tau_inout = np.array(tau_inout)
    res_ssa_inout = np.array(ssa_inout)
    res_g_inout = np.array(g_inout)

    eps = 3.0 * np.finfo(float).tiny

    for igpt in range(1, ngpt + 1):
        for ilay in range(1, nlay + 1):
            for icol in range(1, ncol + 1):
                tau12 = (
                    tau_inout[icol - 1, ilay - 1, igpt - 1]
                    + tau_in[icol - 1, ilay - 1, igpt - 1]
                )
                tauscat12 = (
                    tau_inout[icol - 1, ilay - 1, igpt - 1]
                    * ssa_inout[icol - 1, ilay - 1, igpt - 1]
                    + tau_in[icol - 1, ilay - 1, igpt - 1]
                    * ssa_in[icol - 1, ilay - 1, igpt - 1]
                )

                res_g_inout[icol - 1, ilay - 1, igpt - 1] = (
                    tau_inout[icol - 1, ilay - 1, igpt - 1]
                    * ssa_inout[icol - 1, ilay - 1, igpt - 1]
                    * g_inout[icol - 1, ilay - 1, igpt - 1]
                    + tau_in[icol - 1, ilay - 1, igpt - 1]
                    * ssa_in[icol - 1, ilay - 1, igpt - 1]
                    * g_in[icol - 1, ilay - 1, igpt - 1]
                ) / max(eps, tauscat12)
                res_ssa_inout[icol - 1, ilay - 1, igpt - 1] = tauscat12 / max(
                    eps, tau12
                )
                res_tau_inout[icol - 1, ilay - 1, igpt - 1] = tau12

    py.rte_increment_2stream_by_2stream(*parameters)
    assert (
        np.array_equal(res_tau_inout, tau_inout)
        and np.array_equal(res_ssa_inout, ssa_inout)
        and np.array_equal(res_g_inout, g_inout)
    )


###########################################
## test_rte_increment_2stream_by_nstream ##
###########################################


@pytest.mark.parametrize(
    "parameters,method",
    [
        (
            (
                0,
                0,
                0,
                0,
                np.ones((3, 3, 3)),
                np.ones((3, 3, 3)),
                np.ones((3, 3, 3)),
                np.ones((3, 3, 3)),
                np.ones((3, 3, 3)),
                np.ones((3, 3, 3, 3)),
            ),
            py.rte_increment_2stream_by_nstream,
        )
    ],
)
def test_rte_increment_2stream_by_nstream_dimension_check(parameters, method):
    with pytest.raises(RuntimeError) as excinfo:
        method(*parameters)
    assert str(excinfo.value) == "ncol, nlay, ngpt and nmom must be positive integers"


@pytest.mark.parametrize(
    "parameters,method",
    [
        (
            (
                3,
                3,
                3,
                3,
                np.ones((3, 3)),
                np.ones((3, 3)),
                np.ones((3, 3)),
                np.ones((3, 3)),
                np.ones((3, 3)),
                np.ones((3, 3)),
            ),
            py.rte_increment_2stream_by_nstream,
        )
    ],
)
def test_rte_increment_2stream_by_nstream_array_size_check(parameters, method):
    with pytest.raises(RuntimeError) as excinfo:
        method(*parameters)
    assert str(excinfo.value) == "Invalid size for input arrays"


def test_rte_increment_2stream_by_nstream():

    ncol = 3
    nlay = 4
    ngpt = 5
    nmom = 6
    tau_inout = np.random.rand(ncol, nlay, ngpt)
    ssa_inout = np.random.rand(ncol, nlay, ngpt)
    g_inout = np.random.rand(ncol, nlay, ngpt)
    tau_in = np.random.rand(ncol, nlay, ngpt)
    ssa_in = np.random.rand(ncol, nlay, ngpt)
    p_in = np.random.rand(ncol, nlay, ngpt, nmom)

    expected_tau_inout = np.empty_like(tau_inout)
    expected_ssa_inout = np.empty_like(ssa_inout)
    expected_g_inout = np.empty_like(g_inout)

    eps = 3.0 * np.finfo(float).tiny

    for igpt in range(ngpt):
        for ilay in range(nlay):
            for icol in range(ncol):
                tau12 = tau_inout[icol, ilay, igpt] + tau_in[icol, ilay, igpt]
                tauscat12 = (
                    tau_inout[icol, ilay, igpt] * ssa_inout[icol, ilay, igpt]
                    + tau_in[icol, ilay, igpt] * ssa_in[icol, ilay, igpt]
                )
                expected_g_inout[icol, ilay, igpt] = (
                    tau_inout[icol, ilay, igpt]
                    * ssa_inout[icol, ilay, igpt]
                    * g_inout[icol, ilay, igpt]
                    + tau_in[icol, ilay, igpt]
                    * ssa_in[icol, ilay, igpt]
                    * p_in[icol, ilay, igpt, 0]
                ) / max(tauscat12, eps)
                expected_ssa_inout[icol, ilay, igpt] = tauscat12 / max(eps, tau12)
                expected_tau_inout[icol, ilay, igpt] = tau12

    py.rte_increment_2stream_by_nstream(
        ncol, nlay, ngpt, nmom, tau_inout, ssa_inout, g_inout, tau_in, ssa_in, p_in
    )

    assert np.allclose(expected_tau_inout - tau_inout, 0.0)
    assert np.allclose(expected_ssa_inout - ssa_inout, 0.0)
    assert np.allclose(expected_g_inout - g_inout, 0.0)


###########################################
## test_rte_increment_nstream_by_1scalar ##
###########################################


@pytest.mark.parametrize(
    "parameters,method",
    [
        (
            (0, 0, 0, np.ones((3, 3, 3)), np.ones((3, 3, 3)), np.ones((3, 3, 3))),
            py.rte_increment_nstream_by_1scalar,
        )
    ],
)
def test_rte_increment_nstream_by_1scalar_dimension_check(parameters, method):
    with pytest.raises(RuntimeError) as excinfo:
        method(*parameters)
    assert str(excinfo.value) == "ncol, nlay and ngpt must be positive integers"


@pytest.mark.parametrize(
    "parameters,method",
    [
        (
            (3, 3, 3, np.ones((3, 3)), np.ones((3, 3)), np.ones((3, 3))),
            py.rte_increment_nstream_by_1scalar,
        )
    ],
)
def test_rte_increment_nstream_by_1scalar_array_size_check(parameters, method):
    with pytest.raises(RuntimeError) as excinfo:
        method(*parameters)
    assert str(excinfo.value) == "Invalid size for input arrays"


def test_rte_increment_nstream_by_1scalar():
    ncol, nlay, ngpt = 3, 3, 3
    tau_inout = np.random.rand(ncol, nlay, ngpt)
    g_inout = np.random.rand(ncol, nlay, ngpt)
    tau_in = np.random.rand(ncol, nlay, ngpt)

    res_tau_inout = np.array(tau_inout)
    res_g_inout = np.array(g_inout)

    eps = 3.0 * np.finfo(float).eps
    for icol in range(ncol):
        for ilay in range(nlay):
            for igpt in range(ngpt):
                tau12 = tau_inout[icol, ilay, igpt] + tau_in[icol, ilay, igpt]
                res_g_inout[icol, ilay, igpt] = (
                    tau_inout[icol, ilay, igpt]
                    * g_inout[icol, ilay, igpt]
                    / max(tau12, eps)
                )
                res_tau_inout[icol, ilay, igpt] = tau12

    py.rte_increment_nstream_by_1scalar(ncol, nlay, ngpt, tau_inout, g_inout, tau_in)
    assert np.allclose(res_tau_inout, tau_inout) and np.allclose(res_g_inout, g_inout)


###########################################
## test_rte_increment_nstream_by_2stream ##
###########################################


@pytest.mark.parametrize(
    "parameters,method",
    [
        (
            (
                0,
                0,
                0,
                0,
                np.ones((3, 3, 3)),
                np.ones((3, 3, 3)),
                np.ones((3, 3, 3, 3)),
                np.ones((3, 3, 3)),
                np.ones((3, 3, 3)),
                np.ones((3, 3, 3)),
            ),
            py.rte_increment_nstream_by_2stream,
        )
    ],
)
def test_rte_increment_nstream_by_2stream_dimension_check(parameters, method):
    with pytest.raises(RuntimeError) as excinfo:
        method(*parameters)
    assert str(excinfo.value) == "ncol, nlay, ngpt and nmom1 must be positive integers"


@pytest.mark.parametrize(
    "parameters,method",
    [
        (
            (
                3,
                3,
                3,
                3,
                np.ones((3, 3)),
                np.ones((3, 3)),
                np.ones((3, 3)),
                np.ones((3, 3)),
                np.ones((3, 3)),
                np.ones((3, 3)),
            ),
            py.rte_increment_nstream_by_2stream,
        )
    ],
)
def test_rte_increment_nstream_by_2stream_array_size_check(parameters, method):
    with pytest.raises(RuntimeError) as excinfo:
        method(*parameters)
    assert str(excinfo.value) == "Invalid size for input arrays"


def test_rte_increment_nstream_by_2stream():

    ncol = 3
    nlay = 4
    ngpt = 5
    nmom1 = 6
    tau_inout = np.random.rand(ncol, nlay, ngpt)
    ssa_inout = np.random.rand(ncol, nlay, ngpt)
    p_inout = np.random.rand(ncol, nlay, ngpt, nmom1)
    tau_in = np.random.rand(ncol, nlay, ngpt)
    ssa_in = np.random.rand(ncol, nlay, ngpt)
    g_in = np.random.rand(ncol, nlay, ngpt)

    expected_tau_inout = np.empty_like(tau_inout)
    expected_ssa_inout = np.empty_like(ssa_inout)
    expected_phase_inout = np.empty_like(p_inout)

    eps = 3.0 * np.finfo(float).tiny

    temp_moms = np.zeros(nmom1, dtype=float)

    for icol in range(ncol):
        for ilay in range(nlay):
            for igpt in range(ngpt):
                tau12 = tau_inout[icol, ilay, igpt] + tau_in[icol, ilay, igpt]
                tauscat12 = (
                    tau_inout[icol, ilay, igpt] * ssa_inout[icol, ilay, igpt]
                    + tau_in[icol, ilay, igpt] * ssa_in[icol, ilay, igpt]
                )

                temp_moms[0] = g_in[icol, ilay, igpt]
                for imom in range(1, nmom1):
                    temp_moms[imom] = temp_moms[imom - 1] * g_in[icol, ilay, igpt]

                p_a = (
                    tau_inout[icol, ilay, igpt]
                    * ssa_inout[icol, ilay, igpt]
                    * p_inout[icol, ilay, igpt, 0:nmom1]
                )
                p_b = (
                    tau_in[icol, ilay, igpt]
                    * ssa_in[icol, ilay, igpt]
                    * temp_moms[0:nmom1]
                )
                p_c = (p_a + p_b) / max(tauscat12, eps)

                expected_phase_inout[icol, ilay, igpt, 0:nmom1] = p_c
                expected_ssa_inout[icol, ilay, igpt] = tauscat12 / max(eps, tau12)
                expected_tau_inout[icol, ilay, igpt] = tau12

    py.rte_increment_nstream_by_2stream(
        ncol, nlay, ngpt, nmom1, tau_inout, ssa_inout, p_inout, tau_in, ssa_in, g_in
    )

    assert np.allclose(expected_tau_inout - tau_inout, 0.0)
    assert np.allclose(expected_ssa_inout - ssa_inout, 0.0)
    assert np.allclose(expected_phase_inout - p_inout, 0.0)


###########################################
## test_rte_increment_nstream_by_nstream ##
###########################################


@pytest.mark.parametrize(
    "parameters,method",
    [
        (
            (
                0,
                0,
                0,
                0,
                0,
                np.ones((3, 3, 3)),
                np.ones((3, 3, 3)),
                np.ones((3, 3, 3, 3)),
                np.ones((3, 3, 3)),
                np.ones((3, 3, 3)),
                np.ones((3, 3, 3, 3)),
            ),
            py.rte_increment_nstream_by_nstream,
        )
    ],
)
def test_rte_increment_nstream_by_nstream_dimension_check(parameters, method):
    with pytest.raises(RuntimeError) as excinfo:
        method(*parameters)
    assert (
        str(excinfo.value)
        == "ncol, nlay, ngpt, nmom1 and nmom2 must be positive integers"
    )


@pytest.mark.parametrize(
    "parameters,method",
    [
        (
            (
                3,
                3,
                3,
                3,
                3,
                np.ones((3, 3)),
                np.ones((3, 3)),
                np.ones((3, 3)),
                np.ones((3, 3)),
                np.ones((3, 3)),
                np.ones((3, 3)),
            ),
            py.rte_increment_nstream_by_nstream,
        )
    ],
)
def test_rte_increment_nstream_by_nstream_array_size_check(parameters, method):
    with pytest.raises(RuntimeError) as excinfo:
        method(*parameters)
    assert str(excinfo.value) == "Invalid size for input arrays"


def test_rte_increment_nstream_by_nstream():
    ncol = 3
    nlay = 4
    ngpt = 5
    nmom1 = 6
    nmom2 = 7
    tau_inout = np.random.rand(ncol, nlay, ngpt)
    ssa_inout = np.random.rand(ncol, nlay, ngpt)
    p_inout = np.random.rand(ncol, nlay, ngpt, nmom1)
    tau_in = np.random.rand(ncol, nlay, ngpt)
    ssa_in = np.random.rand(ncol, nlay, ngpt)
    p_in = np.random.rand(ncol, nlay, ngpt, nmom2)

    expected_tau_inout = np.empty_like(tau_inout)
    expected_ssa_inout = np.empty_like(ssa_inout)
    expected_phase_inout = np.empty_like(p_inout)

    mom_lim = min(nmom1, nmom2)

    eps = 3.0 * np.finfo(np.float64).tiny

    for icol in range(ncol):
        for ilay in range(nlay):
            for igpt in range(ngpt):
                tau12 = tau_inout[icol, ilay, igpt] + tau_in[icol, ilay, igpt]

                tauscat12_a = tau_inout[icol, ilay, igpt] * ssa_inout[icol, ilay, igpt]
                tauscat12_b = tau_in[icol, ilay, igpt] * ssa_in[icol, ilay, igpt]
                tauscat12 = tauscat12_a + tauscat12_b

                p_a = tauscat12_a * p_inout[icol, ilay, igpt, 0:mom_lim]
                p_b = tauscat12_b * p_in[icol, ilay, igpt, 0:mom_lim]
                p = (p_a + p_b) / max(tauscat12, eps)

                expected_ssa_inout[icol, ilay, igpt] = tauscat12 / max(eps, tau12)
                expected_tau_inout[icol, ilay, igpt] = tau12
                expected_phase_inout[icol, ilay, igpt, 0:mom_lim] = p

    py.rte_increment_nstream_by_nstream(
        ncol,
        nlay,
        ngpt,
        nmom1,
        nmom2,
        tau_inout,
        ssa_inout,
        p_inout,
        tau_in,
        ssa_in,
        p_in,
    )
    assert np.allclose(expected_tau_inout - tau_inout, 0.0)
    assert np.allclose(expected_ssa_inout - ssa_inout, 0.0)
    assert np.allclose(expected_phase_inout - p_inout, 0.0)
