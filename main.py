from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import pandas as pd
from werkzeug.utils import secure_filename
from langchain_experimental.agents import create_csv_agent
from langchain_community.chat_models import ChatOpenAI
from apikey import OPENAI_API_KEY

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # Ensure the file path is in the form uploads/filename
        return jsonify({"message": "File uploaded successfully", "file_path": file_path.replace('\\', '/')})
    else:
        return jsonify({"message": "Invalid file type. Only CSV files are allowed."})

@app.route("/api/query", methods=["POST"])
def query():
    data = request.json
    user_input = data.get("query")
    file_path = data.get("file_path")
    
    if not file_path or not os.path.exists(file_path):
        return jsonify({"response": "No CSV file found. Please upload a CSV file first."})
    
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    df = pd.read_csv(file_path)
    gpt_4_agent = create_csv_agent(ChatOpenAI(), file_path, verbose=True, allow_dangerous_code=True)
    response = gpt_4_agent.run(user_input)
    
    if "N/A" in response:
        response = "Please ask a question related to the data."

    if "NaN" in response:
        reponse = response.replace("NaN", " ")
    
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)
