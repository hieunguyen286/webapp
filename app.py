from flask import Flask, render_template, request
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import heapq

app = Flask(__name__)

# Download NLTK resources (if not already downloaded)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize_text():
    text = request.form['text']
    
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    word_frequencies = {}
    for word in nltk.word_tokenize(text):
        if word.lower() not in stop_words:
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    # Calculate weighted frequencies
    maximum_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / maximum_frequency)

    # Calculate sentence scores based on word frequencies
    sentence_scores = {}
    for sentence in sentences:
        for word, frequency in word_frequencies.items():
            if word in sentence.lower():
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = frequency
                else:
                    sentence_scores[sentence] += frequency

    # Get the summary with the top N sentences
    summary_sentences = heapq.nlargest(5, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)

    return render_template('index.html', text=text, summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
