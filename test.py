from collision_avoidance_environment.env.environment import Environment
import numpy as np

def main():
    env = Environment(config={'render_mode': 'human'})
    env.reset()

    for _ in range(60):
        action = {'agent_0': np.array([10,0]), 'agent_1': np.array([10,0])}
        obs, rew, _, _, info = env.step(action)
        env.render()

if __name__ == '__main__':
    main()