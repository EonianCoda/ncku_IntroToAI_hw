import json
import random
import sys


MAX_INT = sys.maxsize
def assign_array(target, source):
    for i in range(len(target)):
        target[i] = source[i]

def get_range(num_genes:int):
    start_idx = random.randint(0, num_genes)
    end_idx = random.randint(0, num_genes)
    if start_idx > end_idx:
        tmp = start_idx
        start_idx = end_idx
        end_idx = tmp
    return start_idx, end_idx

class Problem:
    def __init__(self, input):
        self.input = input
        self.numTasks = len(input)
    def cost(self, ans):
        totalTime = 0
        for task, agent in enumerate(ans):
            totalTime += self.input[task][agent]
        return totalTime

class Mutation(object):
    def __call__(self, p1: int, c1: int, population: list):
        raise NotImplementedError
    
class Inversion_mutation(Mutation):
    def __call__(self, p1: int, c1: int, population: list):
        num_genes = len(population[0])
        start_idx, end_idx = get_range(num_genes) 
        for i in range(num_genes):
            if(i >= start_idx and i < end_idx):
                reverse_idx = (end_idx - 1) - (i - start_idx)
                population[c1][i] = population[p1][reverse_idx]
            else:
                population[c1][i] = population[p1][i]

class Swap_mutation(Mutation):
    def __call__(self, p1: int, c1: int, population: list):
        num_genes = len(population[0])
        num_swap = 1
        # Copy parent
        for i in range(num_genes):
            population[c1][i] = population[p1][i]
        for _ in range(num_swap):
            from_idx = random.randint(0, num_genes - 1)
            to_idx = random.randint(0, num_genes - 1)
            if from_idx == to_idx:
                continue
            tmp = population[c1][to_idx]
            population[c1][to_idx] = population[c1][from_idx]
            population[c1][from_idx] = tmp

class Scramble_mutation(Mutation):
    def __call__(self, p1: int, c1: int, population: list):
        num_genes = len(population[0])
        start_idx, end_idx = get_range(num_genes) 
        # Copy parent
        for i in range(num_genes):
            population[c1][i] = population[p1][i]
        random.shuffle(population[c1][start_idx:end_idx])

class Selection(object):
    def __call__(self, fitness_list:list, num_select:int):
        raise NotImplementedError

class Wheel_selection(Selection):
    """Roulette Wheel Selection
    """
    def __call__(self, fitness_list:list, num_select:int):
        indices = [i for i in range(len(fitness_list))]
        
        selected_indices = random.choices(indices, weights=fitness_list, k=num_select)
        return selected_indices
    
class Deterministic_selection(Selection):
    def __call__(self, fitness_list:list, num_select:int):
        num_items = len(fitness_list)
        sorted_indices = sorted(range(num_items), key=lambda k : fitness_list[k])
        sorted_indices.reverse()

        return sorted_indices[:num_select]

class GA(object):
    def __init__(self, pop_size : int = 100,
                        iter_times : int = 300, 
                        cross_ratio : float = 0.2,
                        mut_ratio : float = 0.1,
                        least_fitness_factor : float = 0.3,
                        selection_method : Selection = Wheel_selection,
                        mut_method : Mutation = Inversion_mutation,
                        ):
        self.iter_times = iter_times
        self.least_fitness_factor = least_fitness_factor
        self.pop_size = pop_size # population size
        self.cross_size = int(self.pop_size * cross_ratio) # crossover size
        # Cross(over) size should be even
        if self.cross_size % 2 != 0:
            self.cross_size += 1
        self.mut_size = int(self.pop_size * mut_ratio) # mutation size
        self.total_size = self.pop_size + self.cross_size + self.mut_size
        # For generating random indices
        self.index_template = [i for i in range(self.total_size)]
        # Set mutation and selection methods
        self.mut_method = mut_method()
        self.selection_method = selection_method()

    def initialize(self, problem : Problem):
        self.problem = problem
        self.num_genes = self.problem.numTasks
        # init population
        self.population = [self.gen_random_indices(self.num_genes) for _ in range(self.total_size)]
        self.seleted_parents = [[0] * self.num_genes for _ in range(self.pop_size)]
        self.objetive = [0] * self.total_size
        self.fitness = [0] * self.total_size
        self.best_fitness = -MAX_INT
        self.best_fitness_ans = None
        self.best_obj = MAX_INT
        self.best_obj_ans = None
    def gen_random_indices(self, length):
        indices = self.index_template[:length].copy()
        random.shuffle(indices)
        return indices

    def do_cross_over(self):
        indices = self.gen_random_indices(self.pop_size)
        for i in range(0, self.cross_size, 2):
            p1_idx = indices[i] # parent1
            p2_idx = indices[i + 1] # parent2
            c1_idx = self.pop_size + i
            c2_idx = self.pop_size + i + 1
            self.partially_mapped_crossover(p1_idx, p2_idx, c1_idx, c2_idx)

    def partially_mapped_crossover(self, p1, p2, c1, c2):
        """
        Args:
            p1: the index of parent1
            p2: the index of parent1
            c1: the index of offspring1
            c2: the index of offspring1
        """
        start_idx, end_idx = get_range(self.num_genes) 
        mapping1 = [-1] * self.num_genes
        mapping2 = [-1] * self.num_genes
        for i in range(start_idx, end_idx):
            # swap
            v1 = self.population[p1][i]
            v2 = self.population[p2][i]
            self.population[c1][i] = v2
            self.population[c2][i] = v1
            mapping1[v2] = v1
            mapping2[v1] = v2
        
        for i in range(self.num_genes):
            if i >= start_idx and i < end_idx:
                continue
            v1 = self.population[p1][i]
            v2 = self.population[p2][i]

            while mapping1[v1] != -1:
                v1 = mapping1[v1]
            while mapping2[v2] != -1:
                v2 = mapping2[v2]
            self.population[c1][i] = v1
            self.population[c2][i] = v2
    
    def do_mutation(self):
        # Parents and their children mutates
        indices = self.gen_random_indices(self.pop_size + self.cross_size)
        
        mut_idx = self.pop_size + self.cross_size
        for i in range(self.mut_size):
            self.mut_method(indices[i], mut_idx, self.population)
            mut_idx += 1


    def evaluate_fitness(self):
        min_obj = MAX_INT
        max_obj = -MAX_INT
        # Compute objective values
        for i, ans in enumerate(self.population):
            cost = self.problem.cost(ans)
            if cost < min_obj:
                min_obj = cost
            if cost > max_obj:
                max_obj = cost  
            self.objetive[i] = cost
            if cost < self.best_obj:
                self.best_obj = cost
                self.best_obj_ans = ans.copy()

        range_obj = max_obj - min_obj
        # Compute fitness
        for i, obj in enumerate(self.objetive):
            fitness = max(self.least_fitness_factor * range_obj, pow(10,-5)) + (max_obj - obj)
            self.fitness[i] = fitness
            # Update best fitness and ans
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_fitness_ans = self.population[i].copy()
    
    def do_selection(self):
        num_select = self.pop_size
        seleted_indices = self.selection_method(self.fitness, num_select)

        for i, seleted_idx in enumerate(seleted_indices):
            assign_array(self.seleted_parents[i], self.population[seleted_idx])
        for i in range(num_select):
            assign_array(self.population[i], self.seleted_parents[i])
    def solve(self, problem : Problem):
        self.initialize(problem)
        for _ in range(self.iter_times):
            self.do_cross_over()
            self.do_mutation()
            self.evaluate_fitness()
            self.do_selection()




if __name__ == '__main__':
    # Read json files
    with open("input.json", 'r') as f:
        input_datas = json.load(f)
    solver = GA(pop_size=100, 
                iter_times=300,
                cross_ratio=0.2,
                mut_ratio=0.1,
                least_fitness_factor=0.3,
                selection_method=Wheel_selection,
                mut_method=Inversion_mutation,
                )
    for key, input in input_datas.items():
        problem = Problem(input)
        solver.solve(problem)
        ans = solver.best_obj_ans
        print(key)
        print('Assignment:', ans)
        print('Cost:', problem.cost(ans)) 