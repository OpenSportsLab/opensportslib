import opensportslib

def test_imports():
    """Verify that all main submodules can be imported properly."""
    from opensportslib import apis
    from opensportslib import core
    from opensportslib import datasets
    from opensportslib import metrics
    from opensportslib import models
    
    assert apis is not None
    assert core is not None
    assert datasets is not None
    assert metrics is not None
    assert models is not None
