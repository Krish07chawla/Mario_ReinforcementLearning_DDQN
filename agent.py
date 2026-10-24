import sys, time
from pathlib import Path
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mario_env import make_env
from dqn import MarioNet
from experience_replay import ReplayMemory


def load_params(path="parameters.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)["mario"]


class Mario:
    def __init__(self, p):
        self.p = p
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.action_dim = p["action_dim"]
        self.net = MarioNet(tuple(p["state_shape"]), self.action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.online.parameters(), lr=p["lr"])
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.memory = ReplayMemory(p["memory_size"], self.device)

        self.exploration_rate = p["eps_start"]
        self.curr_step = 0

    # ---------- acting ----------
    def act(self, state):
        """Epsilon-greedy action used during training."""
        if np.random.rand() < self.exploration_rate:
            action = np.random.randint(self.action_dim)
        else:
            action = self.best_action(state)
        self.exploration_rate = max(self.p["eps_min"],
                                    self.exploration_rate * self.p["eps_decay"])
        self.curr_step += 1
        return action

    def best_action(self, state):
        s = torch.from_numpy(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.net(s, model="online")
        return torch.argmax(q, dim=1).item()

    # ---------- Double DQN ----------
    def td_estimate(self, state, action):
        idx = torch.arange(self.p["batch_size"], device=self.device)
        return self.net(state, model="online")[idx, action]

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        idx = torch.arange(self.p["batch_size"], device=self.device)
        best_action = torch.argmax(self.net(next_state, model="online"), dim=1)   # online PICKS
        next_Q = self.net(next_state, model="target")[idx, best_action]            # target EVALUATES
        return (reward + (1 - done) * self.p["gamma"] * next_Q).float()

    def update_Q_online(self, td_est, td_tgt):
        loss = self.loss_fn(td_est, td_tgt)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def learn(self):
        if self.curr_step % self.p["sync_every"] == 0:
            self.sync_Q_target()
        if self.curr_step < self.p["burnin"] or len(self.memory) < self.p["batch_size"]:
            return None, None
        if self.curr_step % self.p["learn_every"] != 0:
            return None, None

        state, next_state, action, reward, done = self.memory.sample(self.p["batch_size"])
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)
        loss = self.update_Q_online(td_est, td_tgt)
        return td_est.mean().item(), loss

    # ---------- checkpoints ----------
    def save(self, path):
        torch.save(dict(model=self.net.state_dict(),
                        optimizer=self.optimizer.state_dict(),
                        exploration_rate=self.exploration_rate,
                        curr_step=self.curr_step), path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.exploration_rate = ckpt["exploration_rate"]
        self.curr_step = ckpt["curr_step"]


def train(resume=False):
    p = load_params()
    env, _ = make_env(p["skip"])
    save_dir = Path(p["save_dir"]); save_dir.mkdir(parents=True, exist_ok=True)
    latest = save_dir / "latest.chkpt"

    mario = Mario(p)
    if resume and latest.exists():
        mario.load(latest)
        print(f"Resumed from {latest} at step {mario.curr_step}")
    print("Device:", mario.device)

    rewards, lengths, losses, qs = [], [], [], []
    t0 = time.time()

    for e in range(p["episodes"]):
        state, info = env.reset()
        ep_r, ep_len, ep_loss, ep_q, n_learn = 0.0, 0, 0.0, 0.0, 0

        while True:
            action = mario.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            mario.memory.cache(state, next_state, action, reward, terminated)
            q, loss = mario.learn()

            ep_r += reward; ep_len += 1
            if loss is not None:
                ep_loss += loss; ep_q += q; n_learn += 1
            state = next_state
            if terminated or truncated or info["flag_get"]:
                break

        rewards.append(ep_r); lengths.append(ep_len)
        losses.append(ep_loss / n_learn if n_learn else 0)
        qs.append(ep_q / n_learn if n_learn else 0)

        if e % p["log_every"] == 0 or e == p["episodes"] - 1:
            m = lambda x: round(float(np.mean(x[-100:])), 3)
            print(f"Episode {e} - Step {mario.curr_step} - Epsilon {mario.exploration_rate:.3f} - "
                  f"Reward {m(rewards)} - Length {m(lengths)} - Loss {m(losses)} - Q {m(qs)} - "
                  f"{time.time()-t0:.0f}s", flush=True)
            mario.save(latest)

    mario.save(latest)
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.5))
    for ax, data, name in zip(axes, [rewards, lengths, losses, qs],
                              ["Reward", "Length", "Loss", "Q-value"]):
        ma = [np.mean(data[max(0, i-99):i+1]) for i in range(len(data))]
        ax.plot(ma); ax.set_title(f"{name} (moving avg 100)")
    plt.tight_layout(); plt.savefig(save_dir / "metrics.png")
    print("Done. Saved", latest, "and metrics.png")


if __name__ == "__main__":
    train(resume="--resume" in sys.argv)
