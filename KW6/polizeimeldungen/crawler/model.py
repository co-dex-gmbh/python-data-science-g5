from dataclasses import dataclass


@dataclass
class PoliceReport:
    date: str = None
    title: str = None
    location: str = None
    link: str = None
    details: str = None  # Das fügen wir später hinzu, wenn wir die Details auslesen
    number: int = None  # Das fügen wir später hinzu, wenn wir die Details auslesen

    # Falls Wir die Ausgabe der Klasse, also print(PoliceReport) anpassen wollen, können wir das mit der __str__ Methode machen
    def __str__(self):
        # return f"{self.date} - {self.title} - {self.location} - {self.link}" # Erst der, später der inkl. Details
        return f"{self.number} - {self.date}\n{self.title} - {self.location} - {self.link}\n{self.details}"
