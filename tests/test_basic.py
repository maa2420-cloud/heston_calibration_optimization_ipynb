def test_import():
    # This checks if the package and its submodules are reachable
    import heston_calib
    from heston_calib import baseline
    from heston_calib import improved
    assert True
