from envs import LunarLanderEnv

def main():
    env = LunarLanderEnv()
    env.reset()
    done = False
    while not done:
        action = env.sample_action()  # Replace with actual control logic
        _, _, done, _ = env.step(action)

if __name__ == "__main__":
    main()
