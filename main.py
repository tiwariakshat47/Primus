from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import re
import datetime
import statistics
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
    gpt_4_agent = create_csv_agent(ChatOpenAI(model="gpt-4"), file_path, verbose=True, allow_dangerous_code=True)
    
    # Determine the type of operation
    classification_prompt = f"Classify the following query as 'math', 'string', 'datetime', 'statistical', 'pattern', 'sql', or 'other':\n\nQuery: {user_input}"
    classification = gpt_4_agent.run(classification_prompt)
    
    if "math" in classification.lower():
        math_prompt = f"Using appropriate libraries like numpy or pandas, perform the following mathematical operation:\n\nQuery: {user_input}"
        response = gpt_4_agent.run(math_prompt)
    elif "string" in classification.lower():
        string_prompt = f"Using appropriate string manipulation libraries, perform the following operation:\n\nQuery: {user_input}"
        response = gpt_4_agent.run(string_prompt)
    elif "datetime" in classification.lower():
        datetime_prompt = f"Using appropriate datetime manipulation libraries, perform the following operation:\n\nQuery: {user_input}"
        response = gpt_4_agent.run(datetime_prompt)
    elif "statistical" in classification.lower():
        statistical_prompt = f"Using appropriate statistical libraries, perform the following operation:\n\nQuery: {user_input}"
        response = gpt_4_agent.run(statistical_prompt)
    elif "pattern" in classification.lower():
        pattern_prompt = f"Using appropriate pattern matching libraries like re, perform the following operation:\n\nQuery: {user_input}"
        response = gpt_4_agent.run(pattern_prompt)
    elif "sql" in classification.lower():
        sql_prompt = f"Using appropriate SQL libraries, perform the following operation:\n\nQuery: {user_input}"
        response = gpt_4_agent.run(sql_prompt)
    else:
        response = gpt_4_agent.run(user_input)
    
    if "N/A" in response:
        response = "Please ask a question related to the data."
    if "NaN" in response:
        response = response.replace("NaN", " ")
    
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)
