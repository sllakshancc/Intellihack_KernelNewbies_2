import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Step 1: Load the Dataset
dataset = pd.read_csv('intent_dataset.csv')

# Step 2: Preprocess the Data
# We'll use a simple bag-of-words representation for text classification
vectorizer = CountVectorizer()

# Step 3: Train a Model
model = Pipeline([
    ('vectorizer', vectorizer),
    ('classifier', MultinomialNB())
])

# Train the model
model.fit(dataset['Text'], dataset['Intent'])

# Step 4: Implement the Classification Function
def classify_intent(text):
    # Predict intent probabilities
    intent_probabilities = model.predict_proba([text])[0]
    # print(intent_probabilities)
    
    # Get the predicted intent
    predicted_intent = model.predict([text])[0]
    # print(predicted_intent)
    
    # Get the confidence score for the predicted intent
    confidence_score = intent_probabilities.max()
    # print(confidence_score)
    
    return predicted_intent, confidence_score

# Step 5: Implement a Fallback Mechanism
def classify_with_fallback(text, threshold=0.7, fallback_response="NLU fallback: Intent could not be confidently determined"):
    predicted_intent, confidence_score = classify_intent(text)
    
    # Check if confidence score meets the threshold
    if confidence_score >= threshold:
        return predicted_intent, confidence_score
    else:
        return fallback_response, None

# Step 6: Test the Model
def main():
    while True:
        # Take input from the user
        user_input = input("Enter your query: ")
        
        # Exit if the user enters 'exit'
        if user_input.lower() == 'exit':
            break
        
        # Classify intent and display the result
        intent, confidence = classify_with_fallback(user_input)
        print(f"Predicted Intent: {intent}, Confidence: {confidence:.2f}" if confidence is not None
              else intent)
        print()

if __name__ == "__main__":
    main()
