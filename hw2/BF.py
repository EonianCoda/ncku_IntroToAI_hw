import sys
import json

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

class BF(object):
    def initialize(self, problem: Problem):
        self.problem = problem
        self.num_worker = problem.numTasks
        self.minima = MAX_INT
        self.best_ans = None

    def solve(self, problem: Problem):
        self.initialize(problem)
        ans = [0] * self.num_worker
        busy = [False] * self.num_worker
        self._solve(ans, busy, 0, 0)

    def _solve(self, ans:list, busy:list, task_idx:int, cost:int):
        """Solve the problem
        This is recursive method for using all combinations to solve problem
        """
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
                self._solve(ans, busy, task_idx + 1, cost + self.problem.input[task_idx][worker_idx])
                busy[worker_idx] = False


if __name__ == '__main__':
    # Read json files
    with open("input.json", 'r') as f:
        input_datas = json.load(f)

    solver = BF()
    for key, input in input_datas.items():
        problem = Problem(input)
        solver.solve(problem)
        ans = solver.best_ans
        print(key)
        print('Assignment:', ans) 
        print('Cost:', problem.cost(ans))

