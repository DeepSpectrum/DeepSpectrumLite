import os
import shutil
import tempfile

def get_tmp_dir():
    tmpdir = tempfile.mkdtemp()
    subdir = os.path.join(tmpdir, "pytest_test_train")
    os.mkdir(subdir)
    return os.path.join(subdir, "")

temp_dir = get_tmp_dir()

def test_train():
    tmpdir = temp_dir
    from deepspectrumlite.__main__ import cli
    from click.testing import CliRunner
    from os.path import dirname, join

    cur_dir = dirname(__file__)
    examples = join(dirname(dirname(cur_dir)), 'examples')

    runner = CliRunner()
    result = runner.invoke(cli,
                           args=[
                               '-vv', 'train',
                               '-d', join(examples, 'audio'),
                               '-md', join(tmpdir, 'output'),
                               '-hc', join(examples, 'hp_config.json'),
                               '-cc', join(examples, 'class_config.json'),
                               '-l', join(examples, 'label.csv')
                           ],
                           catch_exceptions=False)
    assert result.exit_code == 0, f"Exit code for train is not 0 but " + str(result.exit_code)

def test_inference():
    tmpdir = temp_dir
    from deepspectrumlite.__main__ import cli
    from click.testing import CliRunner
    from os.path import dirname, join

    cur_dir = dirname(__file__)
    examples = join(dirname(dirname(cur_dir)), 'examples')

    runner = CliRunner()
    result = runner.invoke(cli,
                           args=[
                               '-vv', 'predict',
                               '-d', join(cur_dir, 'audio'),
                               '-md', join(tmpdir, 'output', 'models', 'densenet_exp', 'densenet121_run_config_0',
                                           'model.h5'),
                               '-hc', join(examples, 'hp_config.json'),
                               '-cc', join(examples, 'class_config.json')
                           ],
                           catch_exceptions=False)
    assert result.exit_code == 0, f"Exit code for predict is not 0 but " + str(result.exit_code)


def test_devel_test():
    tmpdir = temp_dir
    from deepspectrumlite.__main__ import cli
    from click.testing import CliRunner
    from os.path import dirname, join

    cur_dir = dirname(__file__)
    examples = join(dirname(dirname(cur_dir)), 'examples')

    runner = CliRunner()
    result = runner.invoke(cli,
                           args=[
                               '-vv', 'devel-test',
                               '-d', join(examples, 'audio'),
                               '-md', join(tmpdir, 'output', 'models', 'densenet_exp', 'densenet121_run_config_0',
                                           'model.h5'),
                               '-hc', join(examples, 'hp_config.json'),
                               '-cc', join(examples, 'class_config.json'),
                               '-l', join(examples, 'label.csv')
                           ],
                           catch_exceptions=False)
    assert result.exit_code == 0, f"Exit code for devel-test is not 0 but " + str(result.exit_code)


def test_stats():
    tmpdir = temp_dir
    from deepspectrumlite.__main__ import cli
    from click.testing import CliRunner
    from os.path import dirname, join

    runner = CliRunner()
    result = runner.invoke(cli,
                           args=[
                               '-vv', 'stats',
                               '-md', join(tmpdir, 'output', 'models', 'densenet_exp', 'densenet121_run_config_0',
                                           'model.h5')
                           ],
                           catch_exceptions=False)
    assert result.exit_code == 0, f"Exit code for stats is not 0 but " + str(result.exit_code)


def test_convert():
    tmpdir = temp_dir
    from deepspectrumlite.__main__ import cli
    from click.testing import CliRunner
    from os.path import dirname, join

    runner = CliRunner()
    result = runner.invoke(cli,
                           args=[
                               '-vv', 'convert',
                               '-s', join(tmpdir, 'output', 'models', 'densenet_exp', 'densenet121_run_config_0',
                                           'model.h5'),
                               '-d', join(tmpdir, 'output', 'models', 'densenet_exp', 'densenet121_run_config_0',
                                           'converted_model.tflite')
                           ],
                           catch_exceptions=False)
    assert result.exit_code == 0, f"Exit code for convert is not 0 but " + str(result.exit_code)


def test_tflite_stats():
    tmpdir = temp_dir
    from deepspectrumlite.__main__ import cli
    from click.testing import CliRunner
    from os.path import join

    runner = CliRunner()
    result = runner.invoke(cli,
                           args=[
                               '-vv', 'tflite-stats',
                               '-md', join(tmpdir, 'output', 'models', 'densenet_exp', 'densenet121_run_config_0',
                                           'converted_model.tflite')
                           ],
                           catch_exceptions=False)
    assert result.exit_code == 0, f"Exit code for tflite-stats is not 0 but " + str(result.exit_code)


def test_create_preprocessor():
    tmpdir = temp_dir
    from deepspectrumlite.__main__ import cli
    from click.testing import CliRunner
    from os.path import dirname, join

    cur_dir = dirname(__file__)
    examples = join(dirname(dirname(cur_dir)), 'examples')

    runner = CliRunner()
    result = runner.invoke(cli,
                           args=[
                               '-vv', 'create-preprocessor',
                               '-hc', join(examples, 'hp_config.json'),
                               '-d', join(tmpdir, 'output', 'preprocessor.tflite')
                           ],
                           catch_exceptions=False)
    assert result.exit_code == 0, f"Exit code for create_preprocessor is not 0 but " + str(result.exit_code)


def pytest_sessionfinish():
    shutil.rmtree(temp_dir)

# if __name__ == '__main__':
#      print(test_create_preprocessor())