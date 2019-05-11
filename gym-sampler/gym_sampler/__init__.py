from gym.envs.registration import register

register(
    id='sampler-v0',
    entry_point='gym_sampler.envs:SamplerEnv',
)
