import numpy as np
from collections import deque
from mealpy.optimizer import Optimizer

class ARDE(Optimizer):
    def __init__(self, epoch=1000, pop_size=50, memory_size=20, archive_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.memory_size = memory_size
        self.archive_rate = archive_rate

    def initialization(self):
        self.lb = self.problem.lb
        self.ub = self.problem.ub
        self.n_dims = self.problem.n_dims

        self.pop = self.generate_population(self.pop_size)
        self.best = self.get_best_agent(self.pop, minmax="min")
        self.strategy_names = ["rand_1", "best_1", "rand_to_best_1", "best_2", "rand_2"]
        self.q_values = {name: 0.0 for name in self.strategy_names}
        self.memory = {name: deque(maxlen=self.memory_size) for name in self.strategy_names}
        self.archive = deque(maxlen=int(self.archive_rate * self.pop_size))

    def evolve(self, epoch):
        new_pop = []
        for i in range(self.pop_size):
            strategy = self._select_strategy(epsilon=0.1)
            F_i = np.clip(np.random.normal(0.5, 0.3), 0.1, 1.0)
            CR_i = np.clip(np.random.normal(0.9, 0.1), 0, 1)

            new_solution = self._apply_strategy(strategy, i, F_i)
            new_solution = self._crossover(self.pop[i].solution, new_solution, CR_i)
            new_solution = np.clip(new_solution, self.lb, self.ub)
            new_agent = self.generate_agent(new_solution)

            parent_fit = self.pop[i].target.fitness
            new_fit = new_agent.target.fitness

            better_agent = self.get_better_agent(new_agent, self.pop[i], minmax="min")
            new_pop.append(better_agent)

            success = new_fit < parent_fit
            reward = max(0, parent_fit - new_fit)
            self._update_strategy(strategy, reward, success)

            if better_agent.target.fitness < self.best.target.fitness:
                self.archive.append(better_agent)
                self.best = better_agent

        self.pop = new_pop

    def _select_strategy(self, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.choice(self.strategy_names)
        else:
            return max(self.q_values, key=self.q_values.get)

    def _update_strategy(self, strategy, reward, success):
        self.memory[strategy].append(reward)
        avg_reward = np.mean(self.memory[strategy]) if self.memory[strategy] else 0.0
        self.q_values[strategy] = 0.9 * self.q_values[strategy] + 0.1 * avg_reward

    def _apply_strategy(self, strategy, i, F):
        idxs = [j for j in range(self.pop_size) if j != i]
        r = lambda n: np.random.choice(idxs, n, replace=False)
        x = self.pop[i].solution
        x_best = self.best.solution

        if strategy == "rand_1":
            r1, r2, r3 = r(3)
            return self.pop[r1].solution + F * (self.pop[r2].solution - self.pop[r3].solution)

        elif strategy == "best_1":
            r1, r2 = r(2)
            return x_best + F * (self.pop[r1].solution - self.pop[r2].solution)

        elif strategy == "rand_to_best_1":
            r1, r2 = r(2)
            return x + F * (x_best - x) + F * (self.pop[r1].solution - self.pop[r2].solution)

        elif strategy == "best_2":
            r1, r2 = r(2)
            return x + F * (x_best - x) + F * (self.pop[r1].solution - self.pop[r2].solution)

        elif strategy == "rand_2":
            r1, r2, r3, r4, r5 = r(5)
            return self.pop[r1].solution + F * (self.pop[r2].solution - self.pop[r3].solution) + F * (self.pop[r4].solution - self.pop[r5].solution)

    def _crossover(self, target, mutant, CR):
        trial = np.copy(target)
        for j in range(self.n_dims):
            if np.random.rand() < CR or j == np.random.randint(self.n_dims):
                trial[j] = mutant[j]
        return trial

