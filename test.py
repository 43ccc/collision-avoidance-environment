from collision_avoidance_environment.env.environment import Environment
import numpy as np


def main():
    agent_state = {'agent_0': {'position': np.array([250, 250], dtype=np.float32), 'target': np.array([250,400], dtype=np.float32), 'speed': np.array([0], dtype=np.float32), 'direction': np.array([0], dtype=np.float32)}}
    env = Environment(config={'render_mode': 'human'})
    env.reset(options={'agent_state': agent_state})

    for _ in range(60):
        action = {'agent_0': np.array([0,10]), 'agent_1': np.array([0,10])}
        obs, rew, _, _, info = env.step(action)
        env.render()

if __name__ == '__main__':
    main()
