from gym.envs.registration import registry, register, make, spec


register(
    id='HosePullEnv-v0',
    entry_point='gym.envs.custom_env:HosePullEnv'
)