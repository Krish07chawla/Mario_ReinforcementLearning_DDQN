import sys
import imageio
from agent import Mario, load_params
from mario_env import make_env


def evaluate(checkpoint="checkpoints/latest.chkpt", episodes=3, gif="mario_play.gif", eps=0.05):
    p = load_params()
    env, base = make_env(p["skip"])
    mario = Mario(p)
    mario.load(checkpoint)
    print(f"Loaded {checkpoint} (trained steps: {mario.curr_step})")

    import numpy as np
    frames = []
    for ep in range(episodes):
        state, info = env.reset()
        total, steps = 0.0, 0
        while steps < 3000:
            if np.random.rand() < eps:
                action = np.random.randint(p["action_dim"])
            else:
                action = mario.best_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            total += reward; steps += 1
            if ep == 0 and steps % 2 == 0:
                frames.append(base.screen.copy())
            if terminated or truncated or info["flag_get"]:
                break
        print(f"Episode {ep}: reward {total:.0f}, steps {steps}, "
              f"x_pos {info['x_pos']}, reached flag: {info['flag_get']}")

    if frames:
        try:
            imageio.mimsave(gif, frames, fps=15)
        except Exception:
            imageio.mimsave(gif, frames)
        print("Saved", gif)


if __name__ == "__main__":
    evaluate()
