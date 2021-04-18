from gym.envs.registration import register

register(
    id='ArmHitEnv-v1',
    entry_point='arm_env.arm_env:ArmHitEnv',
    reward_threshold=500,
)