# Ex-1-Developing-AI-Agent-with-PEAS-Description
### Name: SUPRAJA B

### Register Number: 2305002026

### Aim:
To find the PEAS description for the given AI problem and develop an AI agent.

### Theory :
PEAS stands for:
'''
P-Performance measure

E-Environment

A-Actuators

S-Sensors
'''

It’s a framework used to define the task environment for an AI agent clearly.

### Pick an AI Problem

```

1. Self-driving car

2. Chess playing agent

3. Vacuum cleaning robot

4. Email spam filter

5. Personal assistant (like Siri or Alexa)
```

### Email spam filter
### Algorithm:
Step 1: Start

Step 2: Collect emails from the inbox (both spam and non-spam).

Step 3: Preprocess the emails

Remove punctuation, stop words, and special characters.

Convert text to lowercase.

Tokenize the words.

Step 4: Extract features

Identify keywords (e.g., “free”, “offer”, “win”, “money”).

Count frequency of words.

Represent data using a feature vector (e.g., bag-of-words or TF-IDF).

Step 5: Train the model

Use a supervised learning algorithm (e.g., Naïve Bayes or SVM).

Input: Labeled dataset (spam / not spam).

Output: Trained spam filter model.

Step 6: Test the model

Provide new/unseen emails to the model.

Predict whether each email is spam or not.

Step 7: Evaluate performance

Measure accuracy, precision, and recall.

Improve model if needed.

Step 8: Classify incoming emails

For each new email → extract features → apply model → label as “spam” or “not spam.”

Step 9: Move spam emails to the spam folder automatically.

Step 10: Stop

### Program:
```
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

emails = [
    "Congratulations! You have won a free lottery ticket",
    "Win a brand new car by clicking this link",
    "Get free coupons now",
    "Let's meet for lunch tomorrow",
    "Your project report has been approved",
    "Are you coming to the meeting today?",
    "You have been selected for a free vacation"
]

labels = [1, 1, 1, 0, 0, 0, 1]   # 1 = spam, 0 = not spam

vectorizer = CountVectorizer()
features = vectorizer.fit_transform(emails)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

test_email = ["Free offer! Click here to win prizes"]
test_features = vectorizer.transform(test_email)
prediction = model.predict(test_features)

if prediction[0] == 1:
    print("Result: Spam Email")
else:
    print("Result: Not Spam Email")

```
### Sample Output:

<img width="381" height="69" alt="image" src="https://github.com/user-attachments/assets/13b74b00-c418-44de-8d16-71944804c451" />

### Result:
