# Collision Avoidance Environment
Basic implementation of a multi agent collision avoidance environment.
The agents have a goal to reach while avoiding colliding with each other. Number of agents, environment size and more can be set. A fixed state can be selected during the reset of the environment.
[alt text](image.png)

## Environment Config
a dict contraining one or more of the following keys and corresponding values can be passed as config during environment initilization.

| Key | Base Value | Description |
|------------------|------------------|------------------|
| num_agents     | 2     | Number of agents     |
| render_mode     | None     | Set to 'human' for vizualisation     |
| environment_size | 500 | Size of the map the agents act on |
| max_timesteps | 1000 | Number of timesteps the agents can take before environment terminates |
| target_reached_threshold | 1 | How far the agent can be from the goal for it to count as reached |
| agent_width | 5 | Width of the agents used in the environment |
| agent_length | 20 | Length of the agents used in the environment |

## Example
test.py shows an example of initilizing and running the environment.