from flask import Flask, render_template, request
from transformers import pipeline
from PyPDF2 import PdfReader

app = Flask(__name__)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    input_text = ""

    if 'file' in request.files:
        file = request.files['file']
        if file.filename.endswith('.pdf'):
            pdf_reader = PdfReader(file)
            input_text = ""
            for page in pdf_reader.pages:
                input_text += page.extract_text()
        elif file.filename.endswith('.txt'):
            input_text = file.read().decode('utf-8')

    if not input_text:
        input_text = request.form['input_text']
    summary_length = request.form['summary_length']
    summary_style = request.form['summary_style']
    
    max_length = 150 if summary_length == "short" else 300
    min_length = 50 

    summary = summarizer(input_text, max_length=max_length, min_length=min_length, do_sample=False)
    
    if summary_style == "bullet":
        formatted_summary = "\n".join([f"- {sentence.strip()}" for sentence in summary[0]['summary_text'].split('.') if sentence])
    else:
        formatted_summary = summary[0]['summary_text']

    return render_template('index.html', summary=formatted_summary, input_text=input_text)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
