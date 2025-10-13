import csv, os
from typing import List, Dict, Any

def export_results_to_csv(results: List[Dict[str, Any]], filename: str) -> None:
    if not results:
        print("No results to export.")
        return

    fitness_keys = list(results[0]["fitness_params"].keys())
    ga_keys = list(results[0]["ga_params"].keys())
    fieldnames = fitness_keys + ga_keys + ["fitness", "time"]

    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            row_data = {**row["fitness_params"], **row["ga_params"], "fitness": row["fitness"], "time": round(row["time"], 2)}
            writer.writerow(row_data)

    print(f"Results exported to {filename}")


def export_result_incremental(result: Dict[str, Any], filename: str) -> None:

    fitness_keys = list(result["fitness_params"].keys())
    ga_keys = list(result["ga_params"].keys())
    fieldnames = fitness_keys + ga_keys + ["fitness", "time"]

    file_exists = os.path.isfile(filename)

    with open(filename, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        row_data = {**result["fitness_params"], **result["ga_params"],
                    "fitness": result["fitness"], "time": round(result["time"], 2)}
        writer.writerow(row_data)


def save_results_to_csv(results: List[Dict], filename: str = "./results.csv") -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not results:
        print("Brak wyników do zapisania")
        return
    keys = results[0].keys()
    with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(keys))
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Wyniki zapisano do pliku CSV: {filename}")
