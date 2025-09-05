# test_sarsa_agent.py

import numpy as np
from wumpus_world import WumpusWorld
import time

def find_sarsa_path():
    env = WumpusWorld()
    try:
        q_table = np.load("q_table_sarsa.npy")
        print("--- Q-table SARSA berhasil dimuat ---")
    except FileNotFoundError:
        print("Error: File 'q_table_sarsa.npy' tidak ditemukan. Jalankan 'train_sarsa.py' terlebih dahulu.")
        return

    def state_to_index(state):
        pos, has_gold = state
        return pos[0] * env.size + pos[1] + (has_gold * (env.size * env.size))

    state = env.reset()
    done = False
    path = [state[0]]
    max_steps = 20
    
    print("\n--- Mencari Jalur Optimal (SARSA) ---")
    env.render()
    time.sleep(1)

    for step in range(max_steps):
        state_idx = state_to_index(state)
        action = np.argmax(q_table[state_idx])
        next_state, reward, done, _ = env.step(action)
        path.append(next_state[0])
        state = next_state
        
        print(f"\nLangkah {step + 1}: Aksi = {env.actions[action]}")
        env.render()
        time.sleep(0.5)

        if done:
            break
            
    print("\n--- Simulasi Selesai ---")
    if env.climbed_out and env.has_gold:
        print("Status: Agen berhasil keluar dengan emas!")
    else:
        print("Status: Agen gagal.")
        
    print("\nJalur yang Ditempuh (SARSA):")
    path_str = " -> ".join([str(p) for p in path])
    print(path_str)

if __name__ == "__main__":
    find_sarsa_path()