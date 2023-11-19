from Scheduling.Scheduling_NSGA_II import NSGA_II, plot_pareto_front, Individual, _minutes_to_time, _initialize_population, BinarySelectionTournament, Mutation, Crossover, FastNonDominatedSort, CrowdingDistance
import random
import matplotlib.pyplot as plt


def main():
    final_population, best = NSGA_II(
        pop_size=100,
        generations=200,
        crossover_prob=0.9,
        mutation_prob=0.1,
        max_blocks_per_individual=10,
        max_block_length=5,
        target_schedule=list(range(300, 1440, 15)) + list(range(0, 70, 15))
    )

    plot_pareto_front(final_population)

if __name__ == "__main__":
    main()
