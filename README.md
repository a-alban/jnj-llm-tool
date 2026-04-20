# JNJ LLM Tool — Financial EDA Platform

A web-based financial data analysis tool that automatically profiles uploaded datasets, applies safe data cleaning, and uses LLMs to generate interactive dashboards and written insights.

## Features
- Upload multiple CSV or Excel files — each gets its own tab
- Automated EDA profiling with 15+ statistical metrics per column
- Policy-gated data cleaning (safe operations only)
- LLM-generated dashboards using Google Gemini 2.5 Flash or Anthropic Claude Sonnet
- Feedback panel to refine dashboards with natural language instructions
- Conversation history kept per file session

## Setup

### 1. Install dependencies
```bash
pip3 install flask google-genai anthropic python-dotenv pandas plotly numpy openpyxl
```

### 2. Add API keys
Create a `.env` file in the project folder:

GEMINI_API_KEY=your_gemini_key_here
ANTHROPIC_API_KEY=your_claude_key_here

### 3. Run the app
```bash
python3 app.py
```

Open your browser and go to `http://127.0.0.1:5000`

## Team
- Andres Alban
- Jose Rodriguez

## Built With
Python, Flask, Plotly, Google Gemini 2.5 Flash, Anthropic Claude Sonnet
