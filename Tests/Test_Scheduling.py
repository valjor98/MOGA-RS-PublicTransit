from Scheduling.Scheduling_NSGA_II import Individual, _minutes_to_time, _initialize_population, BinarySelectionTournament, Mutation, Crossover, FastNonDominatedSort, CrowdingDistance
import random
import matplotlib.pyplot as plt

def main():
    """
    Main execution of the NSGA-II algorithm for the scheduling problem.
    """
    # Parameters
    pop_size = 100
    generations = 200
    crossover_prob = 0.9
    mutation_prob = 0.1
    max_blocks_per_individual = 10  # Maximum number of blocks an individual can have
    max_block_length = 5  # Maximum length of each block

    # Target Schedule
    target_schedule = list(range(300, 1440, 15)) + list(range(0, 70, 15))
    print(f"Target_schedule: {target_schedule}\n")
    formatted_schedule = [_minutes_to_time(minute) for minute in target_schedule]
    print(f"Formatted Target Schedule: {formatted_schedule}\n")
    print(f"{len(target_schedule)} start times in the target schedule\n")


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


    # Visualization at the end
    plt.figure(figsize=(10, 6))
    objectives_1 = [ind.fitness[0] for ind in population]
    objectives_2 = [ind.fitness[1] for ind in population]
    plt.scatter(objectives_1, objectives_2, marker='o')
    plt.title('Objective space')
    plt.xlabel('Objective 1: Uncovered start times')
    plt.ylabel('Objective 2: Number of blocks (drivers)')
    plt.grid(True)
    plt.show()
    

if __name__ == "__main__":
    main()
