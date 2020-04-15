import pytest
import numpy as np



@pytest.mark.parametrize("times, n, expect", [
    ([0.056, 0.067, 0.083, 0.147, 0.233, 0.273, 0.279, 0.286, 0.336, 0.375, 0.388, 0.427, 0.428, 0.438, 0.473], 4, 0.8111),
    ([0.03, 0.07, 0.12], 6, 0.111111)
])
def test_calc_coeff_of_variation(times, n, expect):
    output = calc_coeff_of_variation(times, n)
    assert output == expect





""" @pytest.mark.parametrize("times, window, big_t, n, expect", [
    ([0.01, 0.024, 0.025, 0.0256, 0.031, 0.0314, 0.056], 0.01, 0.06, 3, 0.976),
])
def test_calc_Fano_factor(times, window, big_t, n, expect):
    output = calc_Fano_factor(times, window, big_t, n)
    assert output == expect

@pytest.mark.parametrize("spikes, window,  n, expect", [
    ([0,1,1,0,0,0,1,0,1,0,1], 0.005, 3, 0.4),
    ([0,1,1,0,0,0,1,0,1,0,1], 0.01, 3, 0.133),
    ([0,1,1,0,0,0,1,0,1,0,1], 0.002, 3, 0.545),
    ([0,1,1,0,0,0,1,0,1,0,1,1,0,0], 0.004, 3, 0.476),
    ([1,1,1,0,0,0,1,0,0,0,1,0,0,1,1,0,0,0,1,1,1,0], 0.007, 3, 0.771),

])
def test_calc_Fano_factor2(spikes, window, n, expect):
    output = calc_Fano_factor2(spikes, window, n)
    assert output == expect

@pytest.mark.parametrize("spikes, n, expect", [
    ([0,1,1,0,0,0,1,0,1,0,1,1,0,0], 3, 0.548),
])
def test_calc_coeff_of_variation2(spikes, n, expect):
    output = calc_coeff_of_variation2(spikes, n)
    assert output == expect

@pytest.mark.parametrize("spikes, stimuli, tau, expect", [
    ([0,0,1,0,0,1,1,1,0,1], [20.0,-34.0, 16.0, 12.0, 13.0, -2.0, -6.4, 8.2, -13.1, 12.2], 2, -8.5),
])
def test_get_STA(spikes, stimuli, tau, expect):
    output = get_STA(spikes, stimuli, tau)
    assert output == expect


@pytest.mark.parametrize("spikes, t, expect", [
    ([0,0,1,1,0,1,1,0,0,0,1,0,0], 2, 0.4),
    ([0,0,1,1,0,1,1,0,0,0,1,0,0], 4, 0.2),
    ([0,0,1,1,0,1,1,0,0,0,1,0,0], 6, 0.4),
])
def test_get_autocorr(spikes, t, expect):
    output = get_autocorr(spikes, t)
    assert output == expect

@pytest.mark.parametrize("spikes, stimuli, interval, window, adj, expect", [
    ([0,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,1,0], [20.0, -34.0, 16.0, 12.0, 13.0, -2.0, -6.4, 8.2, -13.1, 12.2, 2.4, -19.3, 0.2, 13.5, 3.2, 17.0, -4.5, -1.1], 6, 10, False, np.array([7.0, 2.8, 10.6, -7.55, 2.9])),
    ([0,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,1,0], [20.0, -34.0, 16.0, 12.0, 13.0, -2.0, -6.4, 8.2, -13.1, 12.2, 2.4, -19.3, 0.2, 13.5, 3.2, 17.0, -4.5, -1.1], 6, 10, True, np.array([0.3, 5.93333333, 7.86666667, -11.46666667, 2])),
])
def test_get_stimulus_pairs(spikes, stimuli, interval, window, adj, expect):
    output = get_stimulus_pairs(spikes, stimuli, interval, window, adj)
    assert np.allclose(output, expect) == True """
