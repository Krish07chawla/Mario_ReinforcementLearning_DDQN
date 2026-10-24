# Mario-playing RL Agent using Double Deep Q-Networks (DDQN)

A Double DQN agent that learns to play Super Mario Bros (World 1-1) from raw pixels.
Based on the PyTorch tutorial "Train a Mario-playing RL Agent", adapted to run on current Google Colab
(Python 3.13) with a plain `deque` replay buffer instead of `torchrl`/`tensordict`.

## Project structure
| File | Purpose |
|---|---|
| `parameters.yaml` | All hyperparameters (episodes, epsilon decay, learning rate, ...) |
| `mario_env.py` | Environment + preprocessing: frame skip (4), grayscale, resize 84x84, stack 4 frames |
| `dqn.py` | `MarioNet`: online and target CNNs |
| `experience_replay.py` | Replay memory (random batch sampling) |
| `agent.py` | `Mario` agent: epsilon-greedy `act`, Double DQN TD estimate/target, training loop |
| `evaluate_model.py` | Loads a checkpoint, plays episodes, saves a GIF |

## How it works
- Actions: 0 = walk right, 1 = jump right.
- Double DQN: the online network picks the best next action, the target network evaluates it.
- Target network synced every 10,000 steps; learning starts after 10,000 steps (burn-in).
- Epsilon decays from 1.0 to 0.1 (`eps_decay: 0.999994`).

## Run
```bash
pip install -r requirements.txt
python agent.py              # train (reads parameters.yaml)
python agent.py --resume     # continue from checkpoints/latest.chkpt
python evaluate_model.py     # watch the trained agent, saves mario_play.gif
```
A GPU is strongly recommended (training was done on a Colab T4).

## Results (3,000 episodes, ~474k steps)
![Mario playing](mario_play.gif)

Evaluation of the final checkpoint (3 episodes, epsilon 0.05):

| Episode | Reward | x_pos reached | Flag |
|---|---|---|---|
| 0 | 602 | 691 | No |
| 1 | 612 | 692 | No |
| 2 | 1304 | 1409 | No |

The agent learned to move right and clear some obstacles but did not finish the level.
The original tutorial suggests ~40,000 episodes for strong performance.
