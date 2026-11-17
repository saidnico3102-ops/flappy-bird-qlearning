from flappy_qlearning import FlappyBirdEnv

def test_reset_observacion():
    dx, dy, vel = FlappyBirdEnv(seed=0).reset()
    assert 0 <= dx <= 288 and vel == 0.0

def test_salto_y_gravedad():
    env = FlappyBirdEnv(seed=0); env.reset()
    env.step(1); assert env.vel == FlappyBirdEnv.FLAP_V
    env.step(0); assert env.vel == FlappyBirdEnv.FLAP_V + FlappyBirdEnv.GRAVITY

def test_reproducible_por_semilla():
    a, b = FlappyBirdEnv(), FlappyBirdEnv()
    assert a.reset(123) == b.reset(123)
    assert [p[1] for p in a.pipes] == [p[1] for p in b.pipes]

def test_choque_con_el_suelo_penaliza():
    env = FlappyBirdEnv(seed=1); env.reset()
    for _ in range(200):
        _, r, done, info = env.step(0)
        if done: break
    assert done and info["crashed"] and r == -1.0
