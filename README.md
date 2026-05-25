ave as README.md:
markdown# IT Incident & Request Bot

An agentic AI chatbot that classifies IT incidents and service 
requests using a hybrid ML + GPT classification engine.

## What it does

- Accepts natural language descriptions of IT issues
- Classifies them using a trained ML model with GPT fallback
- Retrieves relevant context from a knowledge base
- Suggests troubleshooting fixes before ticket creation
- Creates and tracks support tickets automatically
- Secure user authentication with session management

## Architecture

User describes issue
        ↓
ML Classification (TF-IDF + Random Forest)
        ↓ (fallback if low confidence)
GPT Classification
        ↓
Knowledge Base Context Retrieval
        ↓
Troubleshooting Suggestion
        ↓
Ticket Creation and Tracking

## Architecture Note

ServiceNow APIs were not available during development.
Playwright was used to automate browser-based form submission
as a constraint-driven workaround — demonstrating the intended
end-to-end flow to stakeholders before API access was granted.

## Tech Stack

- Python, Flask, Flask-CORS
- scikit-learn (TF-IDF + Random Forest)
- OpenAI API (GPT for classification and troubleshooting)
- SQLite for ticket and user management
- Playwright for browser automation

## Setup

1. Install dependencies:
   pip install -r requirements.txt

2. Add your OpenAI key:
   cp .env.example .env
   Edit .env with your API key

3. Train the ML model:
   cd scripts
   python train_model.py

4. Run the app:
   python app.py

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /api/signup | POST | Register new user |
| /api/login | POST | User login |
| /api/chat | POST | Send message to chatbot |
| /api/submit | POST | Submit ticket |
| /api/troubleshoot | POST | Get AI fix suggestion |
| /api/tickets | POST | View user tickets |
| /api/logout | POST | Logout |

## Project Structure
Inc_Req_Bot/
├── app.py                    # Main Flask application
├── requirements.txt          # Dependencies
├── .env.example             # Environment variables template
├── .gitignore               # Git ignore rules
├── scripts/
│   └── train_model.py       # ML model training
└── knowledgebase/
└── data/
└── training_data.csv # Sample training data

## Key Design Decisions

- Hybrid classification — ML handles known patterns,
  GPT handles edge cases
- Keyword-based context retrieval from CSV knowledge base
- PBKDF2 password hashing with salt for security
- Token-based session management
