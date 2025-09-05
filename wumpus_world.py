# wumpus_world.py (Versi Perbaikan)

import numpy as np

class WumpusWorld:
    def __init__(self):
        self.size = 4
        # Posisi awal agen: Dokumen [1,1] -> Kode (3,0)
        self.start_pos = (3, 0)
        self.agent_pos = self.start_pos
        
        # --- LOKASI ITEM SESUAI DOKUMEN ---
        # Wumpus: Dokumen [1,3] -> Kode (3,2)
        self.wumpus_pos = (3, 2)
        # Emas: Dokumen [2,3] -> Kode (2,2)
        self.gold_pos = (2, 2)
        # Pits: Dokumen [3,1], [3,3], [4,4] -> Kode (1,0), (1,2), (0,3)
        self.pit_pos = [(1, 0), (1, 2), (0, 3)]
        
        # Status permainan
        self.game_over = False
        self.has_gold = False
        self.climbed_out = False
        
        # Aksi yang bisa dilakukan agen: 0=Atas, 1=Bawah, 2=Kiri, 3=Kanan
        self.actions = ['up', 'down', 'left', 'right']

    def reset(self):
        """Mereset lingkungan ke keadaan awal."""
        self.agent_pos = self.start_pos
        self.game_over = False
        self.has_gold = False
        self.climbed_out = False
        return self.get_state()

    def get_state(self):
        """Mengembalikan representasi state dari agen."""
        return (self.agent_pos, 1 if self.has_gold else 0)

    def step(self, action_index):
        """Agen melakukan satu langkah aksi."""
        if self.game_over:
            return self.get_state(), 0, True, {}

        action = self.actions[action_index]
        new_pos = list(self.agent_pos)

        if action == 'up':
            new_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 'down':
            new_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
        elif action == 'left':
            new_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 'right':
            new_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
            
        self.agent_pos = tuple(new_pos)

        reward = -1
        
        if self.agent_pos in self.pit_pos or self.agent_pos == self.wumpus_pos:
            reward = -1000
            self.game_over = True
            
        if self.agent_pos == self.gold_pos and not self.has_gold:
            reward = 1000
            self.has_gold = True
        
        if self.has_gold and self.agent_pos == self.start_pos:
            self.game_over = True
            self.climbed_out = True

        return self.get_state(), reward, self.game_over, {}

    def render(self):
        """Mencetak kondisi grid saat ini."""
        grid = np.full((self.size, self.size), '_', dtype=str)
        grid[self.start_pos] = 'S'
        grid[self.wumpus_pos] = 'W'
        grid[self.gold_pos] = 'G'
        for p in self.pit_pos:
            grid[p] = 'P'
        
        if grid[self.agent_pos] == '_':
            grid[self.agent_pos] = 'A'
        else:
            grid[self.agent_pos] = 'A+' + grid[self.agent_pos]
            
        print("\n".join(" ".join(row) for row in grid))
        print(f"Agen di: {self.agent_pos}, Punya Emas: {self.has_gold}")