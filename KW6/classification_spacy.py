# imports
import pandas as pd
import spacy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import spacy_tests
import seaborn as sns
import datetime
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression  # Beispiel für einen Klassifikator

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier  # falls du XGBoost verwenden möchtest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, \
    f1_score

from openai import OpenAI
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


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
# nlp = spacy.load("de_core_news_lg")

# Block 01 - Erste Tests mit einzelnen Wörter
def first_spaCy():
    nlp = spacy.load("de_core_news_lg")

    one = "Hallo"
    two = "Hello"
    three = "World"
    four = "Polizei"
    five = "Verbrechen"
    six = "Polizist"
    seven = "Polizistin"

    print(nlp(one).similarity(nlp(two)))
    print(nlp(one).similarity(nlp(three)))
    print(nlp(one).similarity(nlp(one)))
    print(nlp(one).similarity(nlp(one)))
    print(nlp(four).similarity(nlp(seven)))
    print(nlp(six).similarity(nlp(seven)))

    print(nlp("Vektor)").vector)
    print("Vektorenlänge: ", len(nlp("Vektor)").vector))


# Block 01 - Weitere Test mit ganzen Sätzen
# Vektoren werden bei spaCy immer für Wörter gebildet, bei Sätzen wir dann aber der Durchschnittsvektor aller Wörter im Satz genutzt.
def second_spaCy():
    nlp = spacy.load("de_core_news_lg")

    # Beispieltexte
    examples = [
        "Ich liebe es zu programmieren.",
        "Heute ist ein schöner Tag.",
        "Machine Learning ist spannend."
    ]

    # Vektorisierung der Sätze
    vektoren = []
    for example in examples:
        doc = nlp(example)  # Verarbeite den Satz mit spaCy
        vektor = doc.vector  # Extrahiere den Vektor
        vektoren.append(vektor)

    print(vektoren[0])


# Block 01 - spaCy functions
def third_spaCy():
    nlp = spacy.load("de_core_news_lg")

    example_one = "Ich mag es zu programmieren und heiße Marten Borchers und wohne in Hamburg."
    example_two = "Gestern Abend wurden in Gropiusstadt Farbschmierereien an einem Wahlkreisbüro einer Partei bemerkt. Gegen 20:15 Uhr stellten alarmierte Polizeikräfte in der Lipschitzallee die Schriftzüge in deutscher Sprache mit Bezug zum Nahostkonflikt fest. Der Polizeiliche Staatsschutz des Landeskriminalamtes Berlin führt die weiteren Ermittlungen."

    doc = nlp(example_one)

    # 1. Tokenisierung und Part-of-Speech-Tagging https://spacy.io/usage/linguistic-features#pos-tagging
    print("\nPart-of-Speech-Tagging:")
    for token in doc:
        print(token.text + ": " + str(token.pos_))

    # 2. Named Entity Recognition https://spacy.io/usage/linguistic-features#named-entities
    print("\nBenannte Entitäten:")
    for ent in doc.ents:
        print(ent.text + ": " + ent.label_)

    # 3. Abhängigkeitsanalyse https://spacy.io/usage/linguistic-features#pos-tagging
    print("\nAbhängigkeitsanalyse:")
    for token in doc:
        print(token.text + ": " + token.dep_ + " - " + token.head.text)

    # 4. Lemmatisierung https://spacy.io/usage/linguistic-features#lemmatization
    print("\nLemmatisierung:")
    for token in doc:
        print(token.text + ": " + token.lemma_)


# Block 02 - spaCy & practice
def fourth_spaCy():
    # {
    #         "date": "05.02.2025 11:10 Uhr",
    #         "title": "Sachbeschädigung an einem Wahlkreisbüro",
    #         "location": "Neukölln",
    #         "link": "/polizei/polizeimeldungen/2025/pressemitteilung.1528367.php",
    #         "details": "Gestern Abend wurden in Gropiusstadt Farbschmierereien an einem Wahlkreisbüro einer Partei bemerkt. Gegen 20:15 Uhr stellten alarmierte Polizeikräfte in der Lipschitzallee die Schriftzüge in deutscher Sprache mit Bezug zum Nahostkonflikt fest. Der Polizeiliche Staatsschutz des Landeskriminalamtes Berlin führt die weiteren Ermittlungen.\n  ",
    #         "number": "0323"
    # }

    nlp = spacy.load("de_core_news_lg")

    # open json as pandas
    df = pd.read_json('datasets/police_reports.json')
    print(df.columns)
    print(len(df))
    print(df.head(5).to_string())
    print(df.loc[:, ['title', 'details']].head())

    # sucht nach genau dem Wort 'Polizei' - funktioniert auch mit Pattern-Matching/direkten Wortvergleich
    def contains_polizei(sentences):
        for word in sentences:
            if nlp(word).similarity(
                    nlp("Polizei")):  # Kein Grenzwert, dabei exakter Überschneidung die cosine-similarity 1 ergibt
                return True
        return False

    # sucht nach Polizei bezogenen Tätigkeiten, die semantisch über das Wort Polizei bestimmt werden
    def contains_polizeiwork(sentences):
        for word in sentences:
            if nlp(word).similarity(nlp("Polizei")) >= 0.7:  # Grenzwert für die Überschneidung
                return True
        return False

    # sucht nach Wahl bezogenen Tätigkeiten, die semantisch über das Wort Wahl bestimmt werden
    def contains_election(sentences):
        for word in sentences:
            if nlp(word).similarity(nlp("Wahl")) >= 0.7:  # Grenzwert für die Überschneidung
                return True
        return False

    arr_contains_polizei = []
    for row in df[['title', 'details']].itertuples(index=False):
        print(f"title: {row.title}, details: {row.details}")
        if contains_polizei(row.title) or contains_polizei(row.details):
            arr_contains_polizei.append(1)
        else:
            arr_contains_polizei.append(0)

    df['Polizei'] = arr_contains_polizei
    print(df.head(5).to_string())

    df.to_csv('datasets/output_file.csv', index=False)


# Block 02 - spaCy & practice
def fifth_spaCy():
    nlp = spacy.load("de_core_news_lg")

    # open json as pandas
    df = pd.read_json('datasets/police_reports.json')
    print(df.columns)
    print(len(df))
    print(df.head(5).to_string())
    print(df.loc[:, ['title', 'details']].head())

    # sucht nach genau dem Wort 'Polizei' - funktioniert auch mit Pattern-Matching/direkten Wortvergleich
    def contains_polizei(sentences):
        for word in sentences:
            if nlp(word).similarity(
                    nlp("Polizei")):  # Kein Grenzwert, dabei exakter Überschneidung die cosine-similarity 1 ergibt
                return True
        return False

    # sucht nach Polizei bezogenen Tätigkeiten, die semantisch über das Wort Polizei bestimmt werden
    def contains_polizeiwork(sentences):
        for word in sentences:
            if nlp(word).similarity(nlp("Polizei")) >= 0.7:  # Grenzwert für die Überschneidung
                return True
        return False

    # sucht nach Wahl bezogenen Tätigkeiten, die semantisch über das Wort Wahl bestimmt werden
    def contains_election(sentences):
        for word in sentences:
            if nlp(word).similarity(nlp("Wahl")) >= 0.7:  # Grenzwert für die Überschneidung
                return True
        return False

    arr_contains_polizei = []
    for row in df[['title', 'details']].itertuples(index=False):
        print(f"title: {row.title}, details: {row.details}")
        if contains_polizei(row.title) or contains_polizei(row.details):
            arr_contains_polizei.append(1)
        else:
            arr_contains_polizei.append(0)

    df['Polizei'] = arr_contains_polizei
    print(df.head(5).to_string())

    df.to_csv('datasets/output_file_extended.csv', index=False)


# Block 04 - Dataset Citizen Participation
def sixth_spaCy():
    nlp = spacy.load("de_core_news_lg")
    # open csv as pandas
    df = pd.read_csv('datasets/dataset_citizen_participation_training_extended.csv', encoding='utf-8', sep=';')
    print(df.head(5).to_string())

    # Erstelle das Balkendiagramm mit Seaborn
    plt.figure(figsize=(10, 6))
    label_counts = df['label'].value_counts().sort_index()
    bar_plot = sns.barplot(x=label_counts.index, y=label_counts.values)
    new_labels = ["Kultur", "Sport", "Umwelt", "Mobilität", "Öffentliche Dienste ", "Soziales",
                  "Wohnen", "Wirtschaft", "Bildung", "Undefiniert", "Sauberkeit", "Lautstärke", "Erholung"]
    bar_plot.set_xticklabels(new_labels, rotation=45)

    # Titel und Achsenbeschriftungen hinzufügen
    plt.title('Häufigkeit der Labels')
    plt.xlabel('Labels')
    plt.ylabel('Häufigkeit')

    # Diagramm anzeigen
    plt.show()


# remove punctuation
def remove_punctuation(sentences: []):
    # Lade das deutsche spaCy-Modell
    nlp = spacy.load("de_core_news_lg")
    cleaned_sentences = []
    for sentence in sentences:
        doc = nlp(sentence)
        # Erstelle eine Liste von Tokens, die keine Satzzeichen sind
        tokens_without_punctuation = [token.text for token in doc if not token.is_punct]
        # Kombiniere die Tokens zu einem Satz
        cleaned_text = " ".join(tokens_without_punctuation)
        cleaned_sentences.append(cleaned_text)
    return cleaned_sentences


def lemmatize_texts(sentences: list) -> list:
    # Lade das deutsche spaCy-Modell
    nlp = spacy.load("de_core_news_lg")

    lemmatized_sentences = []
    for sentence in sentences:
        doc = nlp(sentence)
        # Erstelle eine Liste von Lemmata der Tokens
        lemmatized_tokens = [token.lemma_ for token in doc]
        # Kombiniere die Lemmata zu einem Satz
        lemmatized_text = " ".join(lemmatized_tokens)
        lemmatized_sentences.append(lemmatized_text)

    return lemmatized_sentences


# remove all classes without at least threshold examples
def threshold_labels(df, threshold=1000):
    label_counts = df['label'].value_counts()  # Zähle die Häufigkeit der Labels
    valid_labels = label_counts[
        label_counts > threshold].index  # Halte nur die Labels mit mehr als 'threshold' Beispielen
    filtered_df = df[df['label'].isin(valid_labels)]  # Filtere den DataFrame
    return filtered_df


# vectorisation with spaCy
def vectorize_spacy(sentences: []):
    # Lade das spaCy Modell https://spacy.io/
    nlp = spacy.load("de_core_news_lg")
    # Verarbeitung mit spaCy
    docs = []
    for sentence in sentences:
        docs.append(nlp(sentence).vector)
    # Rückgabe der Durchschnittsvektoren der Token
    return docs


# vectorisation with GloVe
def vectorize_glove(sentences: []):
    # Lade das GloVe Modell https://nlp.stanford.edu/projects/glove/
    file_path = "glove.840B.300d.txt"
    glove_model = KeyedVectors.load_word2vec_format(file_path, binary=False, no_header=True)
    # Verarbeitung mit spaCy
    vector_list = []
    for sentence in sentences:
        vectors = []
        for word in sentence:
            try:
                vectors.append(glove_model[word])  # Holen des Wortvektors
            except KeyError:
                continue  # Ignoriert, wenn das Wort nicht im Vokabular ist
        # Berechnung des Durchschnittsvektors
        if len(vectors) > 0:
            vector_list.append(np.mean(vectors, axis=0))
        else:
            vector_list.append(
                np.zeros(glove_model.vector_size))  # Rückgabe eines Nullvektors, wenn keine Vektoren vorhanden sind
    # Rückgabe der Durchschnittsvektoren der Token
    return vector_list


# lower case
def lower_case(sentences: []):
    lower_sentences = []
    for word in sentences:
        lower_sentences.append(word.lower())
    return lower_sentences


# vectorisation with word2vec
def vectorize_word2vec(sentences: list) -> list:
    # word2vec model = https://cloud.devmount.de/d2bc5672c523b086/
    # https://devmount.github.io/GermanWordEmbeddings/
    # https://huggingface.co/Word2vec/german_model
    model = KeyedVectors.load_word2vec_format('w2c_german.model', binary=True)
    # Eine Liste, um die Vektoren zu speichern
    vectors = []
    for sentence in sentences:
        # Tokenisiere den Satz
        tokens = sentence.split()  # Einfaches Tokenisieren durch Weißraum
        # Berechne den Vektor für den Satz, indem du den Durchschnitt der Token-Vektoren nimmst
        token_vectors = []
        for token in tokens:
            if token in model:  # Überprüfe, ob das Token im Modell vorhanden ist
                token_vectors.append(model[token])
        if token_vectors:  # Wenn es Token-Vektoren gibt
            sentence_vector = np.mean(token_vectors, axis=0)  # Berechne den Durchschnittsvektor
            vectors.append(sentence_vector)
        else:
            vectors.append(np.zeros(model.vector_size))  # Falls keine Token vektorisiert werden konnten
    return vectors


# many classifiers
def sevensth_spaCy():
    # Öffne die CSV-Datei als Pandas DataFrame
    df = pd.read_csv('datasets/dataset_citizen_participation_training.csv', encoding='utf-8', sep=';')
    # print(df.head(5).to_string())

    print("Preprocessing Start: " + str(datetime.datetime.now()))
    filtered_df = threshold_labels(df, threshold=200)  # optional
    print("Threshold Removal Finished: " + str(datetime.datetime.now()))
    X = remove_punctuation(filtered_df['sentence'].tolist())
    print("Punctuation Finished: " + str(datetime.datetime.now()))
    X = lower_case(X)
    print("Lower Finished: " + str(datetime.datetime.now()))
    X = vectorize_word2vec(X)  # vectorize_glove or vectorize_spacy or
    print("Vectorization Finished: " + str(datetime.datetime.now()))
    y = filtered_df['label']
    print("Preprocessing End: " + str(datetime.datetime.now()))

    # X = df['sentence'].apply(vectorize_text).tolist()  # Vectorize the text data
    # y = df['label']  # Labels

    print("X: ", len(X))
    print("y: ", len(y))

    # Split the data into training and test sets (80% training, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("X_train: ", len(X_train))
    print("X_test: ", len(X_test))
    print("y_train: ", len(y_train))
    print("y_test: ", len(y_test))

    # Decision Tree Classifier trainieren
    print("Training Start: " + str(datetime.datetime.now()))
    # clf = DecisionTreeClassifier(random_state=42)
    # clf = RandomForestClassifier(random_state=42)
    clf = SVC(random_state=42)
    # clf = LogisticRegression(random_state=42)
    # clf = GradientBoostingClassifier(random_state=42)
    # clf = XGBClassifier(random_state=42)
    # clf = KNeighborsClassifier()
    clf.fit(list(X_train), y_train)
    print("Training End: " + str(datetime.datetime.now()))

    # Vorhersagen auf Testdaten machen
    y_pred = clf.predict(list(X_test))

    # Ergebnisse evaluieren
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')  # für mehrere Klassen
    recall = recall_score(y_test, y_pred, average='weighted')  # für mehrere Klassen
    f1 = f1_score(y_test, y_pred, average='weighted')  # für mehrere Klassen

    # Konfusionsmatrix berechnen
    confusion = confusion_matrix(y_test, y_pred)

    # Ergebnisse ausgeben
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", confusion)
    print(classification_report(y_test, y_pred))


# bert - DO NOT USE THIS FUNCTION - MEMORY ERROR
def eights_spaCy():
    # Öffne die CSV-Datei als Pandas DataFrame
    df = pd.read_csv('datasets/dataset_citizen_participation_training_extended.csv', encoding='utf-8', sep=';')
    # print(df.head(5).to_string())

    print("Preprocessing Start: " + str(datetime.datetime.now()))
    filtered_df = threshold_labels(df, threshold=200)  # optional
    print("Threshold Removal Finished: " + str(datetime.datetime.now()))
    X = remove_punctuation(filtered_df['sentence'].tolist())
    print("Punctuation Finished: " + str(datetime.datetime.now()))
    y = filtered_df['label']
    print("Preprocessing End: " + str(datetime.datetime.now()))

    print("X: ", len(X))
    print("y: ", len(y))

    # Lade den BERT-Tokenizer und das Modell
    tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')  # Verwende ein deutsches Modell
    model = BertModel.from_pretrained('bert-base-german-cased')

    # Tokeniziere und konvertiere Texte zu BERT-Eingaben
    inputs = tokenizer(X, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # Berechne die Embeddings, indem du das BERT-Modell durchläufst
    with torch.no_grad():
        outputs = model(**inputs)

    # Die Kontextualisierten Embeddings sind im letzten Hidden Layer
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # CLS-Token verwenden

    # Splitte die Daten in Training und Test
    X_train, X_test, y_train, y_test = train_test_split(embeddings, y, test_size=0.2, random_state=42)

    # Trainiere den Klassifikator
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Mache Vorhersagen
    y_pred = clf.predict(X_test)

    # Evaluiere die Vorhersagen
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    # sixth_spaCy()
    sevensth_spaCy()
    # eights_spaCy()
