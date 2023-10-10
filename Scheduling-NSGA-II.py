import random

class Individual:
    def __init__(self, blocks):
        self.blocks = blocks  # List of lists, where each sublist is a block of start times
        self.fitness = []  # List to store the objective values
        self.front = 0
        self.crowding_distance = 0

    def evaluate_objectives(self, target_schedule):
        # Flatten the blocks to get all start times
        all_start_times = [time for block in self.blocks for time in block]

        # Objective 1: Number of uncovered start times
        uncovered_times = sum(1 for t in target_schedule if t not in all_start_times)

        # Objective 2: Number of blocks (drivers)
        num_blocks = len(self.blocks)

        self.fitness = [uncovered_times, num_blocks]


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
    offspring = []
    while len(offspring) < len(population):
        candidates = random.sample(population, 2)
        if candidates[0].front < candidates[1].front:
            winner = candidates[0]
        elif candidates[0].front > candidates[1].front:
            winner = candidates[1]
        else:
            winner = candidates[0] if candidates[0].crowding_distance > candidates[1].crowding_distance else candidates[1]
        offspring.append(winner)
    return offspring


def Crossover(parent1, parent2):
    # Single point crossover for the starting times
    crossover_point = random.randint(0, len(parent1.schedule))
    child1_schedule = parent1.schedule[:crossover_point] + parent2.schedule[crossover_point:]
    child2_schedule = parent2.schedule[:crossover_point] + parent1.schedule[crossover_point:]

    # Create the child individuals and return
    return Individual(child1_schedule), Individual(child2_schedule)

def main():
    """
    Main execution of the NSGA-II algorithm for the scheduling problem.
    """
    # Parameters
    pop_size = 100
    generations = 500
    crossover_prob = 0.9
    mutation_prob = 0.1
    max_blocks_per_individual = 10  # Maximum number of blocks an individual can have
    max_block_length = 5  # Maximum length of each block

    # Target Schedule
    target_schedule = list(range(300, 1440, 15)) + list(range(0, 70, 15))

    # Initialize the population
    population = initialize_population(pop_size, target_schedule, max_blocks_per_individual, max_block_length)

    # Evaluate objectives for the initial population
    for individual in population:
        individual.evaluate_objectives(target_schedule)
    
    for generation in range(generations):
        # Selection
        offspring = BinarySelectionTournament(population)

        # Crossover and Mutation
        for i in range(0, len(offspring) - 1, 2):  # ensuring we have pairs of offspring
            if random.random() < crossover_prob:
                offspring[i], offspring[i + 1] = Crossover(offspring[i], offspring[i + 1])
            Mutation(offspring[i], mutation_prob, target_schedule, max_blocks_per_individual)
            Mutation(offspring[i + 1], mutation_prob, target_schedule, max_blocks_per_individual)

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


if __name__ == "__main__":
    main()
