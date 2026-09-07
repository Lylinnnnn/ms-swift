import time
from types import SimpleNamespace
from unittest.mock import patch

from swift.trainers.patcher import add_train_message


def test_train_progress_metrics_are_numeric_scalars():
    logs = {}
    state = SimpleNamespace(global_step=25, max_steps=100, max_memory=0)
    with patch('swift.trainers.patcher.get_max_reserved_memory', return_value=0):
        add_train_message(logs, state, time.time() - 10, 20)

    assert logs['global_step/max_steps'] == 0.25
    for key in ('global_step/max_steps', 'elapsed_time', 'remaining_time', 'train_speed(s/it)'):
        assert isinstance(logs[key], (int, float))
