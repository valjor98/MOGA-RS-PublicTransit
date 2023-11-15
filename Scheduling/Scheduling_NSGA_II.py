import random
import copy
import matplotlib.pyplot as plt


class Individual:
    def __init__(self, blocks):
        self.blocks = blocks  # List of lists, where each sublist is a block of start times
        self.fitness = []  # List to store the objective values
        self.front = 0
        self.crowding_distance = 0
        self.route = [] # List to store the list of node is of a route

    def _calculate_route_cost(self, node_attributes):
        # Calculate the cost of the route based on node attributes
        weight1 = 0.1
        weight2 = 0.2
        cost = 0
        for node_id in self.route:
            node = node_attributes[node_id]
            cost += node.priority * weight1 + node.zone_type * weight2
        return cost

    def evaluate_objectives(self, target_schedule, node_attributes):
        # Flatten the blocks to get all start times
        all_start_times = [time for block in self.blocks for time in block]

        # Objective 1: Number of uncovered start times
        uncovered_times = sum(1 for t in target_schedule if t not in all_start_times)

        # Objective 2: Number of blocks (drivers)
        num_blocks = len(self.blocks)

        # Objective 3: Route Cost
        self.route_cost = self._calculate_route_cost(node_attributes)

        self.fitness = [uncovered_times, num_blocks]

def _minutes_to_time(minutes):
    """
    Convert minutes after midnight to hour:minute format.
    
    Parameters:
        - minute: Minutes after midnight format
        
    Returns:
        - A string in hour:minute format
    """
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02}:{mins:02}"


def _generate_random_block(target_schedule, max_block_length):
    """
    Helper function to generate 
    
    Parameters:
        - target_schedule: list of the times an ideal line should cover
        - max_block_length: maximum length a single block should have

    Returns:
        - block: the set of stops a single bus makes

    """
    # Randomly choose a starting point and a length for the block
    start_index = random.randint(0, len(target_schedule) - 1)
    block_length = random.randint(1, max_block_length)
    
    # Extract the block of start times
    block = target_schedule[start_index : start_index + block_length]
    
    # Handle wrapping around for cyclic schedules
    if start_index + block_length > len(target_schedule):
        block += target_schedule[:block_length - len(block)]
    
    return block

def _initialize_population(pop_size, target_schedule, max_blocks_per_individual, max_block_length):
    """
    Helper function to generate an initial population of individual solutions.

    Parameters:
        - pop_size: size of the desired population
        - target_schedule: list of the times an ideal line should cover
        - max_blocks_per_individual: maximum number of blocks an individual should have
        - max_block_length: maximum length a single block should have
    Returns:
        - population: a list containing individual solutions
    """
    population = []
    for _ in range(pop_size):
        individual_schedule = [_generate_random_block(target_schedule, max_block_length) for _ in range(random.randint(1, max_blocks_per_individual))]
        population.append(Individual(individual_schedule))
    return population

def BinarySelectionTournament(population):
    """
    Gets two random samples from the populations and sees which one has a better front or crowding distance

    Parameters:
        - population: the population of individuals

    Returns:
        - a list of better suited individuals
    """
    offspring = []
    while len(offspring) < len(population):
        candidates = random.sample(population, 2)
        if candidates[0].front < candidates[1].front:
            winner = candidates[0]
        elif candidates[0].front > candidates[1].front:
            winner = candidates[1]
        else:
            winner = candidates[0] if candidates[0].crowding_distance > candidates[1].crowding_distance else candidates[1]
        #offspring.append(winner)
        offspring.append(copy.deepcopy(winner))
    return offspring


def Crossover(parent1, parent2):
    """
    Makes a crossover between two parents to create two offsprings

    Parameter:
        - parent1: an instance of an Individual solution 
        - parent2: an instance of an Individual solution  

    Returns:
        - Two new offspring individuals which contain attributes of the two parents. 
    """
    crossover_point = random.randint(0, len(parent1.blocks))
    child1_blocks = parent1.blocks[:crossover_point] + parent2.blocks[crossover_point:]
    child2_blocks = parent2.blocks[:crossover_point] + parent1.blocks[crossover_point:]

    return Individual(child1_blocks), Individual(child2_blocks)



def Mutation(individual, prob, target_schedule, max_blocks_per_individual, max_block_length):
    """
    Creates a mutation in an individual based on probability

    Parameters:
        - individual: an instance of a possible solution
        - prob: probability of a mutation
        - target_schedule: list of the times an ideal line should cover
        - max_blocks_per_individual: maximum number of blocks an individual should have
        - max_block_length: maximum length a single block should have

    Returns:
        - Null, creates a mutation within the individual received
    """
    if random.random() < prob:
        mutation_type = random.choice(['add', 'remove', 'modify'])

        # Add a new block to the individual's blocks
        if mutation_type == 'add' and len(individual.blocks) < max_blocks_per_individual:
            new_block = _generate_random_block(target_schedule, max_block_length)
            individual.blocks.append(new_block)

        # Remove a random block from the individual's blocks
        elif mutation_type == 'remove' and len(individual.blocks) > 1:  # Ensuring we don't have empty schedules
            individual.blocks.remove(random.choice(individual.blocks))

        # Modify a random block in the individual's blocks
        elif mutation_type == 'modify':
            idx = random.randrange(len(individual.blocks))
            new_block = _generate_random_block(target_schedule, max_block_length)
            individual.blocks[idx] = new_block


def Dominate(ind1, ind2):
    """
    Checks if individual one dominates individual two

    Parameters:
        - ind1: the first Individual
        - ind2: the second Individual

    Returns:
        - boolean vlaue of whether individual one dominated individual two
    """
    not_worse = all(x <= y for x, y in zip(ind1.fitness, ind2.fitness))
    better = any(x < y for x, y in zip(ind1.fitness, ind2.fitness))
    return not_worse and better

def FastNonDominatedSort(population):
    """
    Rank the population based on non-domination levels. The first front contains non-dominated individuals
    The second front contains individuals dominated by the individuals from the first front and so on

    Parameters:
        - population: the population of individuals

    Returns:
        - A list of lists, containing the individuals that make up each front
    """
    fronts = [[]]
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
            fronts[0].append(p)
    
    i = 0
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
    """
    It measures how close an individual is to its neighbors. It helps in maintaining diversity in the population
    
    Parameters:
        - front: list containing the individuals that belong to a specific front
    
    Returns:
        - null, changes the value crowding_distance of each individual within the received front
    """
    l = len(front)
    for ind in front:
        ind.crowding_distance = 0

    for m in range(len(front[0].fitness)):
        front = sorted(front, key=lambda x: x.fitness[m])
        front[0].crowding_distance = float('inf')
        front[-1].crowding_distance = float('inf')
        f_max = front[-1].fitness[m]
        f_min = front[0].fitness[m]
        if f_max == f_min:  # Prevents division by zero
            for i in range(1, l - 1):
                front[i].crowding_distance += 0
        else:
            for i in range(1, l - 1):
                front[i].crowding_distance += (front[i + 1].fitness[m] - front[i - 1].fitness[m]) / (f_max - f_min)




def NSGA_II(pop_size, generations, crossover_prob, mutation_prob, max_blocks_per_individual, max_block_length, target_schedule):
    """
    Main execution of the NSGA-II algorithm for the scheduling problem.

    Parameters:
        - pop_size: Size of the population.
        - generations: Number of generations to run.
        - crossover_prob: Probability of crossover.
        - mutation_prob: Probability of mutation.
        - max_blocks_per_individual: Maximum number of blocks per individual.
        - max_block_length: Maximum length of each block.
        - target_schedule: List of target start times.
        - Final population and other metrics of interest

    Returns:
        - 
    """

    # Initialize the population
    population = _initialize_population(pop_size, target_schedule, max_blocks_per_individual, max_block_length)

    # Evaluate objectives for the initial population
    for individual in population:
        individual.evaluate_objectives(target_schedule)

    # Initial status print
    best_initial = min(population, key=lambda x: sum(x.fitness))
    worst_initial = max(population, key=lambda x: sum(x.fitness))
    print(f"Initial best objectives: {best_initial.fitness}")
    print(f"Initial worst objectives: {worst_initial.fitness}\n")
    
    for generation in range(generations):
        # Selection
        offspring = BinarySelectionTournament(population)

        # Crossover and Mutation
        for i in range(0, len(offspring) - 1, 2):  # ensuring we have pairs of offspring
            if random.random() < crossover_prob:
                offspring[i], offspring[i + 1] = Crossover(offspring[i], offspring[i + 1])
            Mutation(offspring[i], mutation_prob, target_schedule, max_blocks_per_individual, max_block_length)
            Mutation(offspring[i + 1], mutation_prob, target_schedule, max_blocks_per_individual, max_block_length)

        # Objective Function Evaluation for offspring
        for individual in offspring:
            individual.evaluate_objectives(target_schedule)

        # Combine parent and offspring populations
        combined = population + offspring

        # Fast non-dominated sort
        fronts = FastNonDominatedSort(combined)

        # Assign crowding distance
        for front in fronts:
            CrowdingDistance(front)

        # Select the next generation based on Pareto front and crowding distance
        next_gen_population = []
        for front in fronts:
            if len(next_gen_population) + len(front) <= len(population):
                next_gen_population.extend(front)
            else:
                sorted_front = sorted(front, key=lambda x: (-x.front, -x.crowding_distance))  # select based on front and crowding distance
                next_gen_population.extend(sorted_front[:len(population) - len(next_gen_population)])
                break

        population = next_gen_population

        # Status print at the end of the generation
        best_current = min(population, key=lambda x: sum(x.fitness))
        print(f"Generation {generation + 1}: Best objectives - {best_current.fitness}")
        first_front = [ind for ind in population if ind.front == 0]
        print(f"Number of individuals in the first Pareto front: {len(first_front)}\n")

    return population, best_current

    
def plot_pareto_front(objectives_1, objectives_2):
    """
    Plots the Pareto front of the objectives

    Parameters:
        - objectives_1: First objective values of the population.
        - objectives_2: Second objective values of the population.
    """

    plt.figure(figsize=(10, 6))
    plt.scatter(objectives_1, objectives_2, marker='o')
    plt.title('Objective space')
    plt.xlabel('Objective 1: Uncovered start times')
    plt.ylabel('Objective 2: Number of blocks (drivers)')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    final_population, best = NSGA_II(
        pop_size=100,
        generations=200,
        crossover_prob=0.9,
        mutation_prob=0.1,
        max_blocks_per_individual=10,
        max_block_length=5,
        target_schedule=list(range(300, 1440, 15)) + list(range(0, 70, 15))
    )
    objectives_1 = [ind.fitness[0] for ind in final_population]
    objectives_2 = [ind.fitness[1] for ind in final_population]
    plot_pareto_front(objectives_1, objectives_2)

