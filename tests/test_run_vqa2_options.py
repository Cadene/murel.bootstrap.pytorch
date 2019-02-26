import pytest
import os
import sys
from bootstrap.run import run
from bootstrap.lib.options import Options

option_names = [
    'attention',
    'murel'
]

def reset_options_instance():
    Options._Options__instance = None
    sys.argv = [sys.argv[0]] # reset command line args

@pytest.mark.parametrize('option_name', option_names)
def test_run_vqa2_options(option_name):
    reset_options_instance()
    sys.argv += [
        '-o', f'murel/options/vqa2/{option_name}.yaml',
        '--exp.dir', f'/tmp/logs/tests/vqa2/{option_name}',
        '--engine.nb_epochs', '1',
        '--engine.debug', 'True',
        '--misc.cuda', 'False',
    ]
    try:
        run()
    except:
        print('Unexpected error:', sys.exc_info()[0])
        assert False
    assert True
