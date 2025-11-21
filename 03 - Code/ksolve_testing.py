"""This file has been created to test the ksolve function within windy and wavy.
    since many of the files within the classes require self or randomised data,
    this function was chosen as it only needed one self which was
    "self.dict_windy_wavy["sea_floor"]" which has been replaced with "sea_floor"
    one thing to mention is that ksolve is an interative method as such rounding
    needs to be done to see whether the an acceptable value is verifiable
    the "solved" values for K were calculated for maple
    """

import numpy as np


def ksolve(sea_floor, freq):
    """calculated the 'k' values required to generated the random wave data

    Args:
        freq (numpy.ndarray): array of freqency data

    Returns:
        k_solution (numpy.ndarray): k values for all the frequency values
    """
    k_solution = np.zeros(len(freq))
    gravity = 9.81

    for index, freq_value in enumerate(freq):
        omega_wave = 2 *np.pi *freq_value

        k_star = 1  # initial test value
        error = 1  # initial value
        while error > 0.0001:
            k_estimate = omega_wave ** 2 /\
            (gravity * np.tanh(k_star * sea_floor))
            error = abs(k_estimate - k_star)
            k_star = k_estimate

        k_solution[index] = k_star

    return k_solution

def test_ksolve():
    """tests the kSolve function with 3 specific tests
    """
    # initial test
    freq_test_1 = np.array([1,2,3,4,5])
    sea_floor_test_1 = 10
    expected_k_1 = np.array([4.024303527, 16.09721411, 36.21873175, 64.38885644, 100.6075882])
    ksolve_test_1 = ksolve(sea_floor_test_1, freq_test_1)
    for test_i, _ in enumerate(freq_test_1):
        assert np.isclose(ksolve_test_1[test_i], expected_k_1[test_i])

    #test 2
    #new test that changes sea_floor but keeps same freq
    freq_test_1 = np.array([1,2,3,4,5])
    sea_floor_test_2 = 20
    expected_k_1 = np.array([4.024303527, 16.09721411, 36.21873175, 64.38885644, 100.6075882])
    ksolve_test_2 = ksolve(sea_floor_test_2, freq_test_1)
    for test_i, _ in enumerate(freq_test_1):
        assert np.isclose(ksolve_test_2[test_i], expected_k_1[test_i])

    #test 3
    #new test that changes freq but keeps seafloor the same
    freq_test_1 = np.array([1.5,2.5,7.7,19.5,101])
    sea_floor_test_2 = 20
    expected_k_2 = np.array([9.054682939, 25.15189705, 238.6009562, 1530.241417, 41051.92028])
    ksolve_test_3 = ksolve(sea_floor_test_2, freq_test_1)
    for test_i, _ in enumerate(freq_test_1):
        assert np.isclose(ksolve_test_3[test_i], expected_k_2[test_i])


if __name__ == '__main__':
    test_ksolve()
