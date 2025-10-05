from models import Chromosome
import pandas as pd
import tkinter as tk

# Naturalny porządek dni tygodnia
DAYS_ORDER = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}

def display_schedule(chromosome: Chromosome, sort: bool = True) -> None:
    if sort:
        chromosome = sorted(
            chromosome,
            key=lambda g: (DAYS_ORDER.get(g.slot.day, 7), g.slot.start_time, g.group)
        )
    
    print(f"{'Day':<3} | {'Time':<11} | {'Group':<6} | {'Teacher':<10} | {'Room':<5} | {'Lesson':<20}")
    print("-" * 80)
    
    for gene in chromosome:
        print(f"{gene.slot.day:<3} | "
              f"{gene.slot.start_time}-{gene.slot.end_time:<5} | "
              f"{gene.group:<6} | "
              f"{gene.lesson.teacher:<10} | "
              f"{gene.room:<5} | "
              f"{gene.lesson.name:<20}")

def display_schedule_table(chromosome: Chromosome) -> None:
    # Sort days naturalnie
    days = sorted(set(gene.slot.day for gene in chromosome), key=lambda d: DAYS_ORDER.get(d, 7))
    hours = sorted(set(gene.slot.start_time for gene in chromosome))

    table = pd.DataFrame("", index=hours, columns=days)

    for gene in chromosome:
        content = f"{gene.group}: {gene.lesson.name} ({gene.room})"
        table.at[gene.slot.start_time, gene.slot.day] += content + "\n"

    print(table)

def show_schedule_gui(chromosome: Chromosome) -> None:
    """
    Displays the chromosome as a timetable in a Tkinter window.
    Rows = time slots, Columns = days.
    """
    # Sort days naturalnie
    days = sorted(set(g.slot.day for g in chromosome), key=lambda d: DAYS_ORDER.get(d, 7))
    hours = sorted(set(g.slot.start_time for g in chromosome))

    # Create main window
    root = tk.Tk()
    root.title("Timetable Schedule")

    # Create header row
    for j, day in enumerate(days):
        tk.Label(root, text=day, borderwidth=1, relief="solid", width=20, bg="lightblue").grid(row=0, column=j+1)

    # Create header column
    for i, hour in enumerate(hours):
        tk.Label(root, text=hour, borderwidth=1, relief="solid", width=10, bg="lightgreen").grid(row=i+1, column=0)

    # Fill table cells
    for i, hour in enumerate(hours):
        for j, day in enumerate(days):
            cell_text = ""
            for gene in chromosome:
                if gene.slot.day == day and gene.slot.start_time == hour:
                    cell_text += f"{gene.group}: {gene.lesson.name} ({gene.room})\n"
            tk.Label(
                root, text=cell_text, borderwidth=1, relief="solid",
                width=25, height=6, justify="left", anchor="nw"
            ).grid(row=i+1, column=j+1, sticky="nsew")

    root.mainloop()
