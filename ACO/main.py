from ant_colony_tuner import display_results, tune_parameters
from data import prepare_data_medium
from visualization import plot_timetable, plot_fitness_progress


def main():
    groups, lessons, rooms, slots = prepare_data_medium()

    n_ants_options = [50, 100, 150]
    n_iterations_options = [100, 200, 300]
    evaporation_options = [0.05, 0.075, 0.1, 0.15, 0.2]

    best_plan, best_score, best_params, best_stats = tune_parameters(
        groups, lessons, rooms, slots,
        n_ants_options, n_iterations_options, evaporation_options
    )

    display_results(best_plan, best_score, best_params, best_stats)

    plot_fitness_progress(best_stats, title="Fitness Progress for Best Parameter Set")
    

if __name__ == "__main__":
    main()
