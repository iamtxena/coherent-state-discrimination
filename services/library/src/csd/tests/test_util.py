import pytest
from csd.util import set_friendly_time

only_seconds = [0.1, 21, 51, 59.9]
seconds_and_minutes = [63.4, 120, 821.9181, 962.5343]
expected_seconds_and_minutes = [
    '1 minute and 3 seconds.',
    '2 minutes and 0 seconds.',
    '13 minutes and 41 seconds.',
    '16 minutes and 2 seconds.',
]
until_hours = [12312.0, ]
expected_until_hours = [
    '3 hours, 25 minutes and 12 seconds.',
    '15 hours, 40 minutes and 41 seconds.',
]


@pytest.mark.parametrize("time_interval", only_seconds)
def test_set_friendly_time_only_seconds(time_interval):
    expected_result = f'{int(time_interval)} seconds.'
    assert set_friendly_time(time_interval) == expected_result


@pytest.mark.parametrize("input",
                         zip(seconds_and_minutes, expected_seconds_and_minutes))
def test_set_friendly_time_seconds_and_minutes(input):
    assert set_friendly_time(input[0]) == input[1]


@pytest.mark.parametrize("input",
                         zip(until_hours, expected_until_hours))
def test_set_friendly_time_until_hours(input):
    assert set_friendly_time(input[0]) == input[1]
