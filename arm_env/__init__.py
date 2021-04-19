from gym.envs.registration import register

register(
    id='ArmHitEnv-v1',
    entry_point='arm_env.arm_env:ArmHitEnv',
    max_episode_steps=10000,
    reward_threshold=500,
)