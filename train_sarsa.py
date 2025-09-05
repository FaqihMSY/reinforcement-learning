# train_sarsa.py

import numpy as np
from wumpus_world import WumpusWorld

def train_sarsa():
    env = WumpusWorld()
    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.999
    num_episodes = 20000

    num_states = env.size * env.size * 2
    q_table = np.zeros((num_states, len(env.actions)))

    def state_to_index(state):
        pos, has_gold = state
        return pos[0] * env.size + pos[1] + (has_gold * (env.size * env.size))

    def choose_action(state_idx):
        if np.random.uniform(0, 1) < epsilon:
            return np.random.choice(len(env.actions))
        else:
            return np.argmax(q_table[state_idx])

    for episode in range(num_episodes):
        state = env.reset()
        state_idx = state_to_index(state)
        
        action = choose_action(state_idx)
        
        done = False
        while not done:
            next_state, reward, done, _ = env.step(action)
            next_state_idx = state_to_index(next_state)
            
            next_action = choose_action(next_state_idx)
            
            old_value = q_table[state_idx, action]
            next_value = q_table[next_state_idx, next_action] 
            
            new_value = old_value + alpha * (reward + gamma * next_value - old_value)
            q_table[state_idx, action] = new_value
            
            state_idx = next_state_idx
            action = next_action

        epsilon = max(0.01, epsilon * epsilon_decay)

        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{num_episodes} selesai. Epsilon: {epsilon:.4f}")

    print("\n--- Pelatihan Selesai ---")
    np.set_printoptions(precision=2)
    print("Q-table final (SARSA) yang dihasilkan:")
    print(q_table)
    
    np.save("q_table_sarsa.npy", q_table)
    print("\nQ-table SARSA berhasil disimpan ke 'q_table_sarsa.npy'")
    
    return q_table

if __name__ == "__main__":
    train_sarsa()