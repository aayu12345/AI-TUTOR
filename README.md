# QUIZIFY:📚AI STUDY BUDDY
 
 ###AI-Powered Tutor & Quiz App
 Project Overview
 The AIPowered Tutor & Quiz App is an educational platform that leverages 
artificial intelligence to provide personalized learning experiences. The application 
allows users to ask questions on various academic subjects and receive tailored 
explanations based on their learning preferences. It also generates quizzes to help 
users test their knowledge.
 🛠
 Tools & Technologies
 FastAPI Backend framework for creating the API endpoints
 Streamlit Frontend framework for building the user interface
 LangChain AI orchestration framework for handling LLM interactions
 OpenAI API Powers the AI tutoring and quiz generation capabilities
 Python Primary programming language
 Pydantic Data validation and settings management
 Uvicorn ASGI server for running the FastAPI application
 Python-dotenv For managing environment variables
 🏗
 Architecture & Components
 Component Architecture
 The application follows a client-server architecture with three main components:
  Streamlit Frontend (app.py)
 1
 Personalised Tutor :
User interface for interacting with the application
 Collects user preferences and questions
 Displays personalized explanations and quizzes
  FastAPI Backend (main.py)
 RESTful API endpoints for tutoring and quiz generation
 Handles request validation and error management
 Integrates with the AI engine
 AI Engine (ai_engine.py)
 Core logic for generating personalized tutoring content
 Handles LLM prompt construction and response parsing
 Includes error handling and fallback mechanisms
