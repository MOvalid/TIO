from ant_colony_tuner import display_results, tune_parameters
from data import prepare_data_medium
from visualization import plot_timetable, plot_fitness_progress, compare_parameter_sets


def main():
    groups, lessons, rooms, slots = prepare_data_medium()

    n_ants_options = [50, 100, 150]
    n_iterations_options = [100, 200, 300]
    evaporation_options = [0.05, 0.075, 0.1, 0.15, 0.2]
    alpha_options = [1.0]
    # alpha_options = [0.5, 1.0, 2.0]
    beta_options = [2.0]
    # beta_options = [1.0, 2.0, 5.0]

    best_plan, best_score, best_params, best_stats, all_results = tune_parameters(
        groups, lessons, rooms, slots,
        n_ants_options, n_iterations_options, evaporation_options, alpha_options, beta_options
    )

    display_results(best_plan, best_score, best_params, best_stats)

    plot_fitness_progress(best_stats, title="Fitness Progress for Best Parameter Set", best_params=best_params)
    # plot_timetable(best_plan, title="Best Timetable")
    compare_parameter_sets(all_results, param_names=["n_ants", "n_iterations", "evaporation_rate", "alpha", "beta"],
                           metric='best_score', show=True)
    

if __name__ == "__main__":
    main()
