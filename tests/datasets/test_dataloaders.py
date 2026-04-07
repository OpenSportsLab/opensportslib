def test_datasets_import():
    import opensportslib.datasets
    assert tuple(opensportslib.datasets.__name__.split('.')) == ('opensportslib', 'datasets')
