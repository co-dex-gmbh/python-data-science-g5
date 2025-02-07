import spacy


# Was ist spaCy? -> https://spacy.io/models/de
# spaCy ist eine beliebte Open-Source-Bibliothek für die Verarbeitung natürlicher Sprache (Natural Language Processing, NLP) in Python. Sie wurde entwickelt, um effizient und benutzerfreundlich zu sein und unterstützt verschiedene Aufgaben der Sprachverarbeitung, wie zum Beispiel:
#
# 1. Tokenisierung: Aufteilen eines Textes in Wörter, Sätze oder andere bedeutungstragende Einheiten.
# 2. Teilen von Sätzen: Identifizieren von Satzgrenzen in einem Text.
# 3. Part-of-Speech-Tagging: Bestimmen der grammatikalischen Kategorie (z. B. Nomen, Verb, Adjektiv) von jedem Wort im Text.
# 4. Named Entity Recognition (NER): Erkennen und Klassifizieren von benannten Entitäten wie Personen, Organisationen und Orten.
# 5. Lemmatisierung: Bestimmen der Grundform (Lemma) von Wörtern.
# 6. Abhängigkeitsanalyse: Analyse der grammatikalischen Struktur von Sätzen und der Beziehungen zwischen Wörtern.
# 7. Vektorisierung: Erstellen von Wort- und Dokumenten-Embeddings für maschinelles Lernen.
#
# spaCy wird häufig in der Forschung und der Industrie eingesetzt und ist bekannt für seine Geschwindigkeit und Effizienz beim Umgang mit großen Textmengen. Die Bibliothek unterstützt auch verschiedene Sprachen und ermöglicht die einfache Integration von bestehenden Modellen und Pipelines.
# Wie können wir spaCy nutzen
# 1. pip install spacy
# 2. python -m spacy download de_core_news_md oder de_core_news_lg
# 3. nlp = spacy.load("de_core_news_lg")

# Block 01 - Erste Tests mit einzelnen Wörter
# Aufgabe 1: Installiert spaCy (pip install spacy in eigener .venv) und ladet mindestens ein Model für die Deutsche Sprache herunter. Welche Modelle gibt es?
def first_spaCy():
    nlp = spacy.load("de_core_news_lg")

    one = "Hallo"
    two = "Hello"
    three = "World"
    four = "Polizei"
    five = "Verbrechen"
    six = "Polizist"
    seven = "Polizistin"

    print(nlp(one).vector)
    print(len(nlp(one).vector))

    print(nlp(one).similarity(nlp(two)))
    print(nlp(six).similarity(nlp(seven)))


# Aufgabe 2: Teste die similarity Funktion mir Wörtern wie „Polizei“, „Polizistin“, „Polizist“, 			„Wahlplakat“, „Diebstahl“ und weiteren. Gebt euch für mindestens ein Wort den Vektor aus und überprüft die Vektorenlänge. Wie lang soll die sein?
def second_spaCy():
    return


# Aufgabe 3: Im Plenum – wählt sinnvolle Kategorien für die Analyse der Polizeimeldungen aus Berlin/Hamburg.
def third_spaCy():
    return


# Aufgabe 4: Analysiert die Polizeimeldungen aus Berlin/Hamburg nach diesen Kategorien und 		speichert diese separat ab, damit die Daten nach den Kategorien gefiltert und 		visualisiert werden können.		Kategorien: Title, Datum, Uhrzeit, Delikt (z. B. Diebstahl, Vandalismus), Ort, Wahl	(z. B. Wahlplakat), Täter (Alter, Geschlecht), Opfer (Alter, Geschlecht, Seniorin) und ggf. weitere.
def fourth_spaCy():
    return


# Aufgabe 5: Visualisiert die Daten. Wie viele Wahlkampfbezogenen Meldungen gibt es?
def fourth_spaCy():
    return


# Aufgabe 6: Das ist sehr aufwendig!?! Wie kann die Filterung der der Texte effizienter gestaltet werden?
def fifth_spaCy():
    return


if __name__ == '__main__':
    first_spaCy()
