def test_train(tmpdir):
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
                               '-l', join(examples, 'label.csv'),
                           ])
    print(result.output)
    assert 'Training finished' in result.output, f"Could not finish training"
    assert 'Model saved to' in result.output, f"Could not save model"
    assert 'Confusion matrix' in result.output, f"Could not compute confusion matrix"
    assert result.exit_code == 0, f"Exit code is not success"

# if __name__ == '__main__':
#     test_train('./tmp/')