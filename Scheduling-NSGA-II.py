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
    population = []
    for _ in range(pop_size):
        individual_schedule = [_generate_random_block(target_schedule, max_block_length) for _ in range(random.randint(1, max_blocks_per_individual))]
        population.append(Individual(individual_schedule))
    return population

"""
# Parameters
pop_size = 100
max_blocks_per_individual = 10  # Maximum number of blocks an individual can have
max_block_length = 5  # Maximum length of each block

population = initialize_population(pop_size, target_schedule, max_blocks_per_individual, max_block_length)
"""