from models import AlgorithmData, Lesson, Slot


def prepare_data_easy() -> AlgorithmData:
    groups = ["A", "B"]
    lessons = [Lesson("Math", "John"), Lesson("Physics", "Anna")]
    rooms = ["R1", "R2"]
    slots = [Slot("Mon", "08:00", "09:00"), Slot("Mon", "09:00", "10:00"), Slot("Tue", "08:00", "09:00")]

    return groups, lessons, rooms, slots


def prepare_data_medium() -> AlgorithmData:
    groups = ["A", "B", "C"]

    lessons = [
        Lesson("Math", "John"),
        Lesson("Physics", "Anna"),
        Lesson("Chemistry", "Mike"),
        Lesson("Biology", "Kate")
    ]

    rooms = ["R1", "R2", "R3"]

    slots = [
        Slot("Mon", "08:00", "09:00"),
        Slot("Mon", "09:00", "10:00"),
        Slot("Mon", "10:00", "11:00"),
        Slot("Tue", "08:00", "09:00"),
        Slot("Tue", "09:00", "10:00"),
        Slot("Tue", "10:00", "11:00")
    ]

    return groups, lessons, rooms, slots


def prepare_data_hard() -> AlgorithmData:
    groups = ["A", "B", "C", "D"]

    lessons = [
        Lesson("Math", "John"),
        Lesson("Physics", "Anna"),
        Lesson("Chemistry", "Mike"),
        Lesson("Biology", "Kate"),
        Lesson("History", "Paul"),
        Lesson("English", "Lucy")
    ]

    rooms = ["R1", "R2", "R3", "R4"]

    slots = [
        Slot("Mon", "08:00", "09:00"),
        Slot("Mon", "09:00", "10:00"),
        Slot("Mon", "10:00", "11:00"),
        Slot("Tue", "08:00", "09:00"),
        Slot("Tue", "09:00", "10:00"),
        Slot("Tue", "10:00", "11:00"),
        Slot("Wed", "08:00", "09:00"),
        Slot("Wed", "09:00", "10:00"),
        Slot("Wed", "10:00", "11:00")
    ]

    return groups, lessons, rooms, slots


def prepare_data_hard2() -> AlgorithmData:
    groups = ["A", "B", "C", "D", "E"]

    lessons = [
        Lesson("Math", "John"),
        Lesson("Physics", "Anna"),
        Lesson("Chemistry", "Mike"),
        Lesson("Biology", "Kate"),
        Lesson("History", "Paul"),
        Lesson("English", "Lucy"),
        Lesson("Computer Science", "Tom"),
        Lesson("Geography", "Emma")
    ]

    rooms = ["R1", "R2", "R3", "R4", "R5"]

    slots = [
        Slot("Mon", "08:00", "09:00"),
        Slot("Mon", "09:00", "10:00"),
        Slot("Mon", "10:00", "11:00"),
        Slot("Mon", "11:00", "12:00"),
        Slot("Tue", "08:00", "09:00"),
        Slot("Tue", "09:00", "10:00"),
        Slot("Tue", "10:00", "11:00"),
        Slot("Wed", "08:00", "09:00"),
        Slot("Wed", "09:00", "10:00"),
        Slot("Wed", "10:00", "11:00"),
        Slot("Thu", "08:00", "09:00"),
        Slot("Thu", "09:00", "10:00"),
        Slot("Thu", "10:00", "11:00")
    ]
    return groups, lessons, rooms, slots
