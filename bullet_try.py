import gym
import arm_env

if __name__ == "__main__":
    env = gym.make("ArmHitEnv-v1")
    env.reset()
    while True:
        action = [0, 0, 0, 0, 0, 0]
        new_obs, reward, done, _ = env.step(action)
        if done:
            env.reset()

