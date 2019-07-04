import numpy as np
import random
import sys
import os
from util.getkey import get_manual_arrow_key

# define the map
MAPS = {
    "7x6": [
        "XOOOOOOOX",
        "XOOOOOOOX",
        "XOOOOOOOX",
        "XOOOOOOOX",
        "XOOOOOOOX",
        "XOOOOOOOX",
        "XXXXXXXXX"
    ],
}

# define some colors for terminal output
RED = "\033[1;31m"
BLUE = "\033[1;34m"
CYAN = "\033[1;36m"
GREEN = "\033[0;32m"
RESET = "\033[0;0m"
BOLD = "\033[;1m"
REVERSE = "\033[;7m"


class Game():
    def __init__(self, map_size='7x6', render=True):
        map_layout = np.asarray(MAPS[map_size], dtype='c')
        self.max_row, self.max_col = 6, 7

        map_layout = map_layout.tolist()
        self.map_layout = np.array([[c.decode('utf-8') for c in line] for line in map_layout])
        self.lastaction = None
        self.player =  True #for random: random.choice([True, False])
        self.token = 'A'
        self.game_over = False
        self.render = render
        self.conv_straight = np.array([1,1,1,1])
        self.conv_cross = np.array([[1,0,0,0],
                                    [0,1,0,0],
                                    [0,0,1,0],
                                    [0,0,0,1]])


        if self.render:
            self.rendering()

    def reset(self):
        '''This method resets the game.'''
        map_layout = np.asarray(MAPS['7x6'], dtype='c')
        map_layout = map_layout.tolist()
        self.map_layout = np.array([[c.decode('utf-8') for c in line] for line in map_layout])
        self.lastaction = None
        self.game_over = False
        self.player = True
        return self.get_state()

    def get_state(self):
        '''Returns the current state of the game'''
        state = self.map_layout[0:6,1:8].copy()
        if self.token == 'A':
            anti_token = 'B'
        else:
            anti_token = 'A'

        state[state == self.token] = 1
        state[state == anti_token] = 2
        state[state == 'O'] = 0
        state = state.astype(np.int8)
        state[state == 2] = -1

        return state.astype(np.float)

    def rendering(self):
        '''This method renders the game in the console. For each step the console will be cleared and then reprinted'''
        os.system('cls||clear')
        if self.lastaction is not None:
            sys.stdout.write(BLUE)
            sys.stdout.write("  ({})\n".format(self.lastaction))
        else:
            sys.stdout.write("\n")

        for idx_r, r in enumerate(self.map_layout):
            for idx_c, c in enumerate(r):
                sys.stdout.write(RESET)
                if c in 'A':
                    sys.stdout.write(RED)
                if c in 'B':
                    sys.stdout.write(GREEN)
                if c in 'O':
                    sys.stdout.write(CYAN)
                sys.stdout.write(c + ' ')
            sys.stdout.write('\n')



    def get_reward(self):
        c_map = self.map_layout.copy()
        c_map[c_map == self.token] = 1
        c_map[c_map!='1'] = 0
        c_map = c_map.astype(np.int8)

        '''Returns the reward of the current player position and sets the game_over boolean.'''
        for row in np.flip(c_map, 0):
            for col in range(0,5):
                result = np.sum(self.conv_straight*row[col:col+4])
                if result == 4:
                    self.game_over = True
                    return (1, -1)

        for col in c_map.transpose():
            for row in range(0,3):
                result = np.sum(self.conv_straight*col[row:row+4])
                if result == 4:
                    self.game_over = True
                    return (1, -1)

        for row in range(0,3):
            for col in range(0,5):
                result_l = np.sum(self.conv_cross*c_map[row:row+4, col:col+4])
                result_r = np.sum(np.flip(self.conv_cross, 1)*c_map[row:row+4, col:col+4])
                if result_r == 4 or result_l == 4:
                    self.game_over = True
                    return (1, -1)

        return (-0.01, 0)

    def perform_action(self, action):
        '''
        Performs the given action in the environment.

        action -- String representation of the action, e.g. left, down, right or up.
        '''
        # sys.stdout.write(self.token)
        if self.player == True:
            # self.render = True
            self.token = 'A'
        else:
            # self.render = True
            self.token = 'B'

        for id, pos in enumerate(self.map_layout[:,action]):
            if pos != 'O':
                if id == 0:
                    # print('Column {:1} is already complete. Pick another column.'.format(action))
                    return self.get_reward(), self.get_state(), self.game_over
                self.map_layout[id-1,action] = self.token
                break

        self.player = not self.player

        self.lastaction = action
        if self.render:
            self.rendering()

        return self.get_reward(), self.get_state(), self.game_over



if __name__ == "__main__":
    game = Game()
    game_over = False
    while not game_over:
        # play game with helper class, which will detect keyboard inputs (numbers 1-7)
        _, _, g = game.perform_action(get_manual_arrow_key())

        game_over = g
