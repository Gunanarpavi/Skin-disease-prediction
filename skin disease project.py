import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data = {
'itching': [1, 0, 1, 0, 1, 1],
'rash': [1, 1, 0, 1, 1, 0],
'skin_peeling': [0, 1, 0, 0, 1, 0],
'dry_skin': [1, 0, 1, 1, 0, 1],
'disease': ['eczema','psoriasis', 'eczema','fungal_infection','psoriasis', 'eczema']
}
df = pd.DataFrame(data)

X = df[['itching','rash','skin_peeling','dry_skin']]
y = df['disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))

def predict_disease(itching, rash, skin_peeling, dry_skin):
  input_data = [[itching, rash, skin_peeling, dry_skin]]
  prediction = model.predict(input_data)
  print(f"Predicted Disease: {prediction[0]}")

  predict_disease(1, 1, 0, 1)
