import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Individual:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.fitness = []
        self.front = 0
        self.crowding_distance = 0

    def evaluate_objectives(self):
        f1 = self.x ** 2 + (self.y - 1) ** 2
        f2 = self.x ** 2 + (self.y + 1) ** 2 + 1
        f3 = (self.x - 1) ** 2 + self.y ** 2 + 2
        self.fitness = [f1, f2, f3]

def Objectives(individual):
    individual.evaluate_objectives()

# gets two random samples from the populations and sees which one has a better front or crowding distance
def BinarySelectionTournament(population):
    offspring = []
    while len(offspring) < len(population):
        candidates = random.sample(population, 2)
        if candidates[0].front < candidates[1].front:
            winner = candidates[0]
        elif candidates[0].front > candidates[1].front:
            winner = candidates[1]
        else:
            winner = candidates[0] if candidates[0].crowding_distance > candidates[1].crowding_distance else candidates[1]
        offspring.append(Individual(winner.x, winner.y))
    return offspring

def Crossover(parent1, parent2):
    alpha = random.random()
    # alpha helps the randomization of obtaining a parent's x and y attributes
    child1_x = alpha * parent1.x + (1 - alpha) * parent2.x
    child2_x = alpha * parent2.x + (1 - alpha) * parent1.x
    alpha = random.random()
    child1_y = alpha * parent1.y + (1 - alpha) * parent2.y
    child2_y = alpha * parent2.y + (1 - alpha) * parent1.y
    return Individual(child1_x, child1_y), Individual(child2_x, child2_y)

def Mutation(individual, prob, xmin, xmax, ymin, ymax):
    if random.random() < prob:
        # Adds/Subtracts 0.1 value from x and y 
        individual.x += random.uniform(-0.1, 0.1)
        individual.y += random.uniform(-0.1, 0.1)
        # Makes sure the values are in-bound within limits
        individual.x = max(min(individual.x, xmax), xmin)
        individual.y = max(min(individual.y, ymax), ymin)

def Dominate(ind1, ind2):
    # Checks that ind1 is not worse than ind2 in all objectives
    not_worse = all(x <= y for x, y in zip(ind1.fitness, ind2.fitness))
    # checks if ind1 is better than ind2 in at least one objective
    better = any(x < y for x, y in zip(ind1.fitness, ind2.fitness))
    return not_worse and better

def FastNonDominatedSort(population):
    fronts = []
    first_front = []
    # selects the first and best pareto front
    for p in population:
        p.domination_count = 0
        p.dominated_solutions = []
        for q in population:
            if Dominate(p, q):
                p.dominated_solutions.append(q)
            elif Dominate(q, p):
                p.domination_count += 1
        if p.domination_count == 0:
            p.front = 0
            first_front.append(p)
    fronts.append(first_front)
    i = 0
    # Gets the rest of the pareto fronts
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in p.dominated_solutions:
                q.domination_count -= 1
                if q.domination_count == 0:
                    q.front = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return fronts[:-1]

def CrowdingDistance(front):
    l = len(front)
    for ind in front:
        ind.crowding_distance = 0
    
    for m in range(len(front[0].fitness)):
        front = sorted(front, key=lambda x: x.fitness[m])
        # asigns infinity to the crowding distance of the boundarie values of the given front
        front[0].crowding_distance = front[-1].crowding_distance = float('inf')
        f_max = front[-1].fitness[m]
        f_min = front[0].fitness[m]
        # for the rests, checks how spread out are each value based on the fitness
        for i in range(1, l - 1):
            front[i].crowding_distance += (front[i + 1].fitness[m] - front[i - 1].fitness[m]) / (f_max - f_min)

def main():
    # Parameters
    P = 100  # size of the parent population
    Q = 100  # size of the offspring population
    R = P + Q  # size of the joint population
    generations = 500  # Changed to 500 for demonstration
    crossover_prob = 0.9
    mutation_prob = 0.1
    xmin, xmax = -5, 5  # limits for x
    ymin, ymax = -5, 5  # limits for y

    # Initialize parent population
    population = [Individual(random.uniform(xmin, xmax), random.uniform(ymin, ymax)) for _ in range(P)]

    # Evaluate objectives for the initial population
    for individual in population:
        Objectives(individual)

    for generation in range(generations):

        # Selection
        offspring = BinarySelectionTournament(population)

        # Crossover
        # parses through 
        for i in range(0, Q, 2):
            if random.random() < crossover_prob:
                offspring[i], offspring[i + 1] = Crossover(offspring[i], offspring[i + 1])

        # Mutation
        # Mutates individuals based on probability
        for individual in offspring:
            Mutation(individual, mutation_prob, xmin, xmax, ymin, ymax)

        # Evaluate objectives for the offspring in the offsprings
        for individual in offspring:
            Objectives(individual)

        # Combine parent and offspring populations
        combined = population + offspring

        # Fast non-dominated sort
        fronts = FastNonDominatedSort(combined)

        # Assign crowding distance
        for front in fronts:
            CrowdingDistance(front)

        # Create new population
        population = []
        for front in fronts:
            if len(population) + len(front) <= P:
                population += front
            else:
                front = sorted(front, key=lambda x: x.crowding_distance, reverse=True)
                population += front[:P - len(population)]
                break

        # Plotting the first Pareto front at intervals of 100 generations and the final generation
        if generation % 100 == 0 or generation == generations - 1:
            if fronts:
                first_pareto_front = fronts[0]
                x = [ind.fitness[0] for ind in first_pareto_front]
                y = [ind.fitness[1] for ind in first_pareto_front]
                z = [ind.fitness[2] for ind in first_pareto_front]
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(x, y, z)
                ax.set_xlabel('Objective 1')
                ax.set_ylabel('Objective 2')
                ax.set_zlabel('Objective 3')
                plt.title(f'Pareto Front at Generation {generation}')
                plt.show()

if __name__ == "__main__":
    main()

