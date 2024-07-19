import os
import random
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import streamlit as st

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Function to load intents from intents.json
def load_intents(file_path):
    with open(file_path) as file:
        intents = json.load(file)
    return intents

# Function to preprocess data and train the model
def train_model(intents_path):
    intents = load_intents(intents_path)

    words = []
    classes = []
    documents = []
    ignore_words = ['?', '!', '.', ',']

    for intent in intents['intents']:
        for example in intent['examples']:
            word_list = nltk.word_tokenize(example)
            words.extend(word_list)
            documents.append((word_list, intent['intent']))
            if intent['intent'] not in classes:
                classes.append(intent['intent'])

    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_words]
    words = sorted(set(words))

    classes = sorted(set(classes))

    training = []
    output_empty = [0] * len(classes)

    for document in documents:
        bag = []
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns if word not in ignore_words]

        for word in words:
            bag.append(1 if word in word_patterns else 0)

        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training, dtype=object)

    train_x = np.array([i[0] for i in training])
    train_y = np.array([i[1] for i in training])

    model = Sequential()
    model.add(Dense(128, input_shape=(len(words),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(classes), activation='softmax'))

    adam = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
    
    model.save('model/chatbot_model.keras')

    # Save words and classes
    with open('model/words_classes.json', 'w') as file:
        json.dump({'words': words, 'classes': classes}, file)

    return model, words, classes

# Utility functions
def clean_up_sentence(sentence):
    ignore_words = ['?', '!', '.', ',']
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words if word not in ignore_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = np.zeros(len(words), dtype=np.float32)
    for w in sentence_words:
        if w in words:
            bag[words.index(w)] = 1
    return bag

def predict_classes(sentence, model, words, classes):
    bow = bag_of_words(sentence, words)
    bow = np.reshape(bow, (1, len(words)))  # Ensure input shape matches model's expectations
    res = model.predict(bow)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return return_list

def get_response(intents_list, intents):
    tag = intents_list[0]['intent']
    list_of_intents = intents['intents']
    for intent in list_of_intents:
        if intent['intent'] == tag:
            return random.choice(intent['responses'])
    return "I don't understand!"

def add_new_intent(user_input, intents_path):
    intents = load_intents(intents_path)
    new_tag = f"new_{len(intents['intents']) + 1}"
    new_intent = {
        "intent": new_tag,
        "examples": [user_input],
        "responses": ["I'm learning something new!"]
    }
    intents['intents'].append(new_intent)
    with open(intents_path, 'w') as file:
        json.dump(intents, file, indent=4)
    return new_tag

# Streamlit application
def main():
    st.title("VISION - AI")
    st.write("I'm recently born to this world so I'm not familiar with all human things.")

    intents_path = 'intents.json'

    if not os.path.exists('model/chatbot_model.keras'):
        st.write("Training the model for the first time...")
        model, words, classes = train_model(intents_path)
        st.write("Model trained successfully!")
    else:
        model = load_model('model/chatbot_model.keras')
        if not os.path.exists('model/words_classes.json'):
            st.write("Training the model for the first time (words/classes)...")
            model, words, classes = train_model(intents_path)
            st.write("Model trained successfully!")
        else:
            with open('model/words_classes.json', 'r') as file:
                data = json.load(file)
                words = data['words']
                classes = data['classes']

    user_input = st.text_input("You:", "")
    if user_input:
        model = load_model('model/chatbot_model.keras')
        intents = load_intents(intents_path)
        intents_list = predict_classes(user_input, model, words, classes)
        if intents_list:
            response = get_response(intents_list, intents)
            st.text_area("Bot:", response, height=100)
        else:
            st.text_area("Bot:", "I don't understand!", height=100)
            st.write("Learning new input...")
            new_tag = add_new_intent(user_input, intents_path)
            st.write(f"New intent added: {new_tag}")
            st.write("The model studies the new data...")
            st.cache_data.clear()
            with st.spinner('Training the model...'):
                model, words, classes = train_model(intents_path)
            st.write("Model retrained successfully!")

if __name__ == '__main__':
    main()
