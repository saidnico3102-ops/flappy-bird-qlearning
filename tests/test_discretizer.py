from flappy_qlearning import CONFIG, StateDiscretizer

def test_total_de_estados():
    n = 1
    for b in CONFIG["bins"]: n *= b
    assert n == 3600

def test_indices_dentro_de_rango_incluso_con_valores_extremos():
    d = StateDiscretizer(CONFIG)
    for obs in [(0, -1e4, -50), (1e4, 1e4, 50), (144, 0, 0)]:
        s = d.discretize(obs)
        assert all(0 <= v < b for v, b in zip(s, CONFIG["bins"]))
