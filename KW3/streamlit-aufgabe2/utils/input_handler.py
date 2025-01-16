

def aggregation_mapping(aggregation):
    if aggregation == "Woche":
        resolution = "W"
    elif aggregation == "Monat":
        resolution = "ME"
    elif aggregation == "Jahr":
        resolution = "Y"
    else:
        resolution = "D"
    return resolution
