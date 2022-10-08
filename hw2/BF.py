import random
import sys

MAX_INT = sys.maxsize
class Problem:
    def __init__(self, input):
        self.input = input
        self.numTasks = len(input)
    def cost(self, ans):
        totalTime = 0
        for task, agent in enumerate(ans):
            totalTime += self.input[task][agent]
        return totalTime

class Exhaustive(object):
    def __init__(self, problem : Problem):
        self.problem = problem
        self.num_worker = problem.numTasks
        self.minima = MAX_INT
        self.best_ans = None
    def get_ans(self):
        ans = [0] * self.num_worker
        busy = [False] * self.num_worker
        self.solve(ans, busy, 0, 0)
        return self.best_ans

    def solve(self, ans:list, busy:list, task_idx:int, cost:int):
        if cost >= self.minima:
            return
        if task_idx == self.num_worker:
            if cost < self.minima:
                self.minima = cost
                self.best_ans = ans.copy()
            return

        for worker_idx in range(self.num_worker):
            if not busy[worker_idx]:
                busy[worker_idx] = True
                ans[task_idx] = worker_idx
                self.solve(ans, busy, task_idx + 1, cost + self.problem.input[task_idx][worker_idx])
                busy[worker_idx] = False


if __name__ == '__main__':
    # import random

    # width = 11
    # input = [[0] * width] * width
    # for i in range(width):
    #     for j in range(width):
    #         input[i][j] = random.random()
        
    input =[[0.43045255, 0.78681387, 0.07514408, 0.72583933, 0.52916145, 0.87483212, 0.34701621],
            [0.68704291, 0.45392742, 0.46862110, 0.67669006, 0.23817468, 0.87520581, 0.67311418],
            [0.38505150, 0.05974168, 0.11388629, 0.28978058, 0.66089373, 0.92592403, 0.70718757],
            [0.24975701, 0.16937649, 0.42003672, 0.88231235, 0.74635725, 0.59854858, 0.88631100], 
            [0.64895582, 0.58909596, 0.99772334, 0.85522575, 0.33916707, 0.72873479, 0.26826203],
            [0.47939038, 0.88484586, 0.05122520, 0.83527995, 0.37219939, 0.20375257, 0.50482283],
            [0.58926554, 0.45176739, 0.25217475, 0.83548120, 0.41687026, 0.00293049, 0.23939052]]



    problem = Problem(input)
    solver = Exhaustive(problem)
    ans = solver.get_ans()
    print('Assignment:', ans) # print 出分配結果
    print('Cost:', problem.cost(ans)) # print 出 cost 是多少

