from neuron import train_and_test
import pytest

TRAIN_F = "../data/train.dat"
TEST_F = "../data/test.dat"
# lr, epochs, expected_accuracy
tests = ((0.005, 15, 68),
         (0.01, 50, 76),
         (0.025, 25, 76),
         (0.025, 50, 80),
         )


def test_neuron():
    for test in tests:
        lr, epochs, expected_accuracy = test
        accuracy = train_and_test(TRAIN_F, TEST_F, lr, epochs)
        assert accuracy == pytest.approx(expected_accuracy, rel=0.001)