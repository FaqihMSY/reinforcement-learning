# train_q_learning.py (Versi Perbaikan)

import numpy as np
from wumpus_world import WumpusWorld

def train():
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

    for episode in range(num_episodes):
        state = env.reset()
        state_idx = state_to_index(state)
        done = False
        
        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.choice(len(env.actions))
            else:
                action = np.argmax(q_table[state_idx])
            
            next_state, reward, done, _ = env.step(action)
            next_state_idx = state_to_index(next_state)
            
            old_value = q_table[state_idx, action]
            next_max = np.max(q_table[next_state_idx])
            
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            q_table[state_idx, action] = new_value
            
            state_idx = next_state_idx

        epsilon = max(0.01, epsilon * epsilon_decay)

        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{num_episodes} selesai. Epsilon: {epsilon:.4f}")

    print("\n--- Pelatihan Selesai ---")
    np.set_printoptions(precision=2)
    print("Q-table final (32 states):")
    print(q_table)
    
    np.save("q_table.npy", q_table)
    print("\nQ-table berhasil disimpan ke 'q_table.npy'")
    
    return q_table

if __name__ == "__main__":
    train()