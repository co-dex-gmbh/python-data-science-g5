from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# Relevante Features auswählen und fehlende Werte behandeln
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]
df['Age'].fillna(df['Age'].median(), inplace=True)

# Kategorische Daten in numerische Formate umwandeln
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])

# Features und Zielvariable definieren
X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']

# Daten in Trainings- und Testsets aufteilen
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Naive Bayes Classifier initialisieren und trainieren
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Vorhersagen auf dem Testset treffen
y_pred = nb_classifier.predict(X_test)

# Genauigkeit des Modells bewerten
accuracy = accuracy_score(y_test, y_pred)
print(f'Genauigkeit des Naive Bayes Classifiers: {accuracy:.2f}')
