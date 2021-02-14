from gym.envs.registration import register

register(
    id='Sushi-v0',
    entry_point='sushi.envs:SushiEnv',
)

