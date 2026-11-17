import pytest
from flappy_qlearning import CONFIG, QLearningAgent

def test_inicializacion_optimista():
    assert QLearningAgent(CONFIG).Q.shape == (15, 20, 12, 2) and QLearningAgent(CONFIG).Q.min() == 2.0

def test_regla_de_bellman():
    ag = QLearningAgent(CONFIG); s, s2 = (0, 0, 0), (1, 1, 1)
    ag.update(s, 1, 1.1, s2, terminal=False)
    assert ag.Q[s + (1,)] == pytest.approx(2.0 + 0.1 * (1.1 + 0.95 * 2.0 - 2.0))
    ag.update(s, 0, -1.0, s2, terminal=True)
    assert ag.Q[s + (0,)] == pytest.approx(2.0 + 0.1 * (-1.0 - 2.0))

def test_empate_no_salta():
    assert QLearningAgent(CONFIG).select_action((3, 3, 3), greedy=True) == 0

def test_decaimiento_de_epsilon():
    ag = QLearningAgent(CONFIG); ag.end_episode(919)
    assert ag.eps == pytest.approx(0.01)
