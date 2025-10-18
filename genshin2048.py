from ailib import ailib, to_c_board
import numpy as np
from numpy import log2
import time
class Genshin2048:
    def __init__(self, seed=None):
        self.board=None
        self.move=0
    def init_board(self):
        self.board=np.zeros((4,4), dtype=int)
        self.board[0,0]=2
        self.board[1,2]=2
    def get_board(self):
        return self.board
    def set_board(self,board):
        self.board=np.copy(board)
    def get_move(self):
        board=np.copy(self.board)
        for i in range(4):
            for j in range(4):
                if board[i][j] != 0:
                    board[i][j]=int(log2(board[i][j]))
        return ailib.find_best_move(to_c_board(board))

    @staticmethod
    def move_left_on_board(old_board):
        # board 是 np.ndarray，但逻辑不变（仍按行处理）
        board = np.copy(old_board)
        for r in range(4):
            line = [board[r][0], board[r][1], board[r][2], board[r][3]]
            i = 0
            while i < 3:
                # 找到右边第一个非零元素
                j = i + 1
                while j < 4 and line[j] == 0:
                    j += 1
                if j == 4:  # 右边没有非零元素了
                    break
                if line[i] == 0:
                    # 移动元素
                    line[i] = line[j]
                    line[j] = 0
                    i -= 1  # 重新检查当前位置
                elif line[i] == line[j] and line[i] !=2048:
                    # 合并元素
                    line[i] *= 2
                    line[j] = 0
                i += 1
            # 更新棋盘
            for c in range(4):
                board[r][c] = line[c]
        return board

    def add_tile(self,random=True):
        move = self.move
        empty_tile = []
        new_grid=[]
        if move==0:
            for i in range(4):
                if self.board[3][i]==0:
                    empty_tile.append(i)
            random_index = np.random.choice(empty_tile)
            if random:
                self.board[3][random_index]=2
            else:
                for index in empty_tile:
                    new_grid.append((3,index))
        elif move==1:
            for i in range(4):
                if self.board[0][i]==0:
                    empty_tile.append(i)
            random_index = np.random.choice(empty_tile)
            if random:
                self.board[0][random_index]=2
            else:
                for index in empty_tile:
                    new_grid.append((0,index))
        elif move==2:
            for i in range(4):
                if self.board[i][3]==0:
                    empty_tile.append(i)
            random_index = np.random.choice(empty_tile)
            if random:
                self.board[random_index][3]=2
            else:
                for index in empty_tile:
                    new_grid.append((index,3))
        elif move==3:
            for i in range(4):
                if self.board[i][0]==0:
                    empty_tile.append(i)
            random_index = np.random.choice(empty_tile)
            if random:
                self.board[random_index][0]=2
            else:
                for index in empty_tile:
                    new_grid.append((index,0))
        return new_grid

    def move_tile(self):
        board=np.copy(self.board)
        direction = ["up","down","left","right"][self.move]
        if direction == 'left':
            self.board= self.move_left_on_board(board)
        elif direction == 'right':
            board = np.rot90(self.board, k=2)
            board= self.move_left_on_board(board)
            self.board = np.rot90(board, k=2)
        elif direction == 'down':
            board = np.rot90(self.board, k=3)
            board = self.move_left_on_board(board)
            self.board = np.rot90(board, k=1)
        elif direction == 'up':
            board = np.rot90(self.board, k=1)
            board = self.move_left_on_board(board)
            self.board = np.rot90(board, k=3)

    def run(self):
        self.init_board()
        step=1
        print(f"2048 game design for Genshin Impact web version")
        print("\n")
        print("=" * 5, f"The {step} board ", "=" * 5)
        self.move = self.get_move()
        direction = ["up","down","left","right"]
        while True:
            print(f"#step:{step}, move:{direction[self.move]}")
            self.move_tile()
            self.add_tile()
            print("\n")
            print("=" * 5, f"The {step} board ", "=" * 5)
            self.move = self.get_move()
            if self.move not in [0,1,2,3]:
                break
            step += 1

if __name__=="__main__":
    start_time = time.time()
    g=Genshin2048()
    g.run()
 #    g.init_board()
 #    g.set_board(np.array([[ 2 , 0 , 0 , 2],
 # [ 0 , 0 , 0 , 2],
 # [ 0  ,0,  2 , 8],
 # [ 2,  8, 16,  8]]))
 #    print(g.get_board())
 #    g.move=1
 #    g.move_tile()
 #    print(g.get_board())
    print(f"---运行游戏耗时{time.time() - start_time} seconds ---")
