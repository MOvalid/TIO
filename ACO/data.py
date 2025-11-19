from models import AlgorithmData, Lesson, Slot


def prepare_data_easy() -> AlgorithmData:
    groups = ["A", "B"]
    lessons = [Lesson("Math", "John"), Lesson("Physics", "Anna")]
    rooms = ["R1", "R2"]
    slots = [Slot("Mon", "08:00", "09:00"), Slot("Mon", "09:00", "10:00"), Slot("Mon", "10:00", "11:00"),Slot("Mon", "11:00", "12:00"),
             Slot("Tue", "08:00", "09:00"), Slot("Tue", "09:00", "10:00"), Slot("Tue", "10:00", "11:00"),Slot("Tue", "11:00", "12:00")]

    return groups, lessons, rooms, slots


def prepare_data_medium() -> AlgorithmData:
    groups = ["A", "B", "C"]

    lessons = [
        Lesson("Math", "John"),
        Lesson("Physics", "Anna"),
        Lesson("Chemistry", "Mike"),
        Lesson("Biology", "Kate")
    ]

    rooms = ["R1", "R2", "R3", "R4"]

    slots = [
        Slot("Mon", "07:00", "08:00"),
        Slot("Mon", "08:00", "09:00"),
        Slot("Mon", "09:00", "10:00"),
        Slot("Mon", "10:00", "11:00"),
        Slot("Mon", "11:00", "12:00"),
        Slot("Mon", "12:00", "13:00"),
        Slot("Mon", "13:00", "14:00"),
        Slot("Tue", "07:00", "08:00"),
        Slot("Tue", "08:00", "09:00"),
        Slot("Tue", "09:00", "10:00"),
        Slot("Tue", "10:00", "11:00"),
        Slot("Tue", "11:00", "12:00"),
        Slot("Tue", "12:00", "13:00"),
        Slot("Tue", "13:00", "14:00"),
    ]

    return groups, lessons, rooms, slots
