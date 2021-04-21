def test_cividis(tmpdir):
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
                               '-hc', join(cur_dir, 'config', 'cividis_hp_config.json'),
                               '-cc', join(examples, 'class_config.json'),
                               '-l', join(examples, 'label.csv')
                           ],
                           catch_exceptions=False)
    assert result.exit_code == 0, f"Exit code for train is not 0 but " + str(result.exit_code)

def test_plasma(tmpdir):
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
                               '-hc', join(cur_dir, 'config', 'plasma_hp_config.json'),
                               '-cc', join(examples, 'class_config.json'),
                               '-l', join(examples, 'label.csv')
                           ],
                           catch_exceptions=False)
    assert result.exit_code == 0, f"Exit code for train is not 0 but " + str(result.exit_code)

def test_inferno(tmpdir):
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
                               '-hc', join(cur_dir, 'config', 'inferno_hp_config.json'),
                               '-cc', join(examples, 'class_config.json'),
                               '-l', join(examples, 'label.csv')
                           ],
                           catch_exceptions=False)
    assert result.exit_code == 0, f"Exit code for train is not 0 but " + str(result.exit_code)

def test_magma(tmpdir):
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
                               '-hc', join(cur_dir, 'config', 'magma_hp_config.json'),
                               '-cc', join(examples, 'class_config.json'),
                               '-l', join(examples, 'label.csv')
                           ],
                           catch_exceptions=False)
    assert result.exit_code == 0, f"Exit code for train is not 0 but " + str(result.exit_code)

def test_regression(tmpdir):
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
                               '-hc', join(cur_dir, 'config', 'regression_hp_config.json'),
                               '-cc', join(cur_dir, 'config', 'regression_class_config.json'),
                               '-l', join(cur_dir, 'label', 'regression_label.csv')
                           ],
                           catch_exceptions=False)
    assert result.exit_code == 0, f"Exit code for train is not 0 but " + str(result.exit_code)