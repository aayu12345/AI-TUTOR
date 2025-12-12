from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
import os
from dotenv import load_dotenv
import json
import re
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def get_llm():
    try:
        return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=GROQ_API_KEY

)
    except Exception as e:
        logger.error(f"Error initializing LLM: {str(e)}")
        raise Exception(f"Failed to initialize AI model: {str(e)}")


def generate_tutoring_response(subject, level, question, learning_style, background, language):
    """
    Generate a personalized tutoring response based on user preferences.
    
    Args:
        subject (str): The academic subject
        level (str): Learning level (Beginner, Intermediate, Advanced)
        question (str): User's specific question
        learning_style (str): Preferred learning style (Visual, Text-based, Hands-on)
        background (str): User's background knowledge
        language (str): Preferred language for response
    
    Returns:
        str: Formatted tutoring response
    """
    try:
        # Get LLM instance
        llm = get_llm()
        
        # Construct an effective prompt
        prompt = _create_tutoring_prompt(subject, level, question, learning_style, background, language)
        
        # Generate response with error handling
        logger.info(f"Generating tutoring response for subject: {subject}, level: {level}")
        response = llm.invoke(prompt)
        
        # Post-process the response based on learning style
        return _format_tutoring_response(response.content, learning_style)
        
    except Exception as e:
        logger.error(f"Error generating tutoring response: {str(e)}")
        raise Exception(f"Failed to generate tutoring response: {str(e)}")



def _create_tutoring_prompt(subject, level, question, learning_style, background, language):
    """Helper function to create a well-structured tutoring prompt"""
    
    # Build the prompt with all necessary context and instruction
    prompt = f"""
    You are an expert tutor in {subject} at the {level} level. 
    
    STUDENT PROFILE:
    - Background knowledge: {background}
    - Learning style preference: {learning_style}
    - Language preference: {language}
    
    QUESTION:
    {question}
    
    INSTRUCTIONS:
    1. Provide a clear, educational explanation that directly addresses the question
    2. Tailor your explanation to a {background} student at {level} level
    3. Use {language} as the primary language
    4. Format your response with appropriate markdown for readability
    
    LEARNING STYLE ADAPTATIONS:
    - For Visual learners: Include descriptions of visual concepts, diagrams, or mental models
    - For Text-based learners: Provide clear, structured explanations with defined concepts
    - For Hands-on learners: Include practical examples, exercises, or applications
    
    Your explanation should be educational, accurate, and engaging.
    """
    
    return prompt



def _format_tutoring_response(content, learning_style):
    """Helper function to format the tutoring response based on learning style"""
    
    if learning_style == "Visual":
        return content + "\n\n*Note: Visualize these concepts as you read for better retention.*"
    elif learning_style == "Hands-on":
        return content + "\n\n*Tip: Try working through the examples yourself to reinforce your learning.*"
    else:
        return content
    


def _create_quiz_prompt(subject, level,topic, num_questions):
    """Helper function to create a well-structured quiz generation prompt"""
    
    return f"""
    Create a {level}-level quiz of {subject} on topic {topic} with exactly {num_questions} multiple-choice questions.
    
    INSTRUCTIONS:
    1. Each question should be appropriate for {level} level students
    2. Each question must have exactly 4 answer options (A, B, C, D)
    3. Clearly indicate the correct answer
    4. Cover diverse aspects of {subject} on topic {topic}
    
    FORMAT YOUR RESPONSE AS JSON:
    ```json
    [
        {{
            "question": "Question text",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct_answer": "Option A",
            "explanation": "Brief explanation of why this answer is correct"
        }},
        ...
    ]
    ```
    
    IMPORTANT: Make sure to return valid JSON that can be parsed. Do not include any text outside the JSON array.
    Include a brief explanation for each correct answer.
    """

def _create_fallback_quiz(subject, num_questions):
    """Helper function to create a fallback quiz if parsing fails"""
    
    logger.warning(f"Using fallback quiz for {subject}")
    
    return [
        {
            "question": f"Sample {subject} question #{i+1}",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct_answer": "Option A",
            "explanation": "This is a fallback explanation."
        }
        for i in range(num_questions)
    ]



def _validate_quiz_data(quiz_data):
    """Helper function to validate quiz data structure"""
    
    if not isinstance(quiz_data, list):
        raise ValueError("Quiz data must be a list of questions")
    
    for question in quiz_data:
        if not isinstance(question, dict):
            raise ValueError("Each quiz item must be a dictionary")
            
        if not all(key in question for key in ["question", "options", "correct_answer"]):
            raise ValueError("Each quiz item must have question, options, and correct_answer")
            
        if not isinstance(question["options"], list) or len(question["options"]) != 4:
            raise ValueError("Each question must have exactly 4 options")
        


def _parse_quiz_response(response_content, subject, num_questions):
    """Helper function to parse and validate the quiz response"""
    
    try:
        num_questions = int(num_questions)
        # Try to find JSON content using regex
        json_match = re.search(r'```json\s*(\[[\s\S]*?\])\s*```', response_content)
        
        if json_match:
            # Extract JSON from code block
            quiz_json = json_match.group(1)
        else:
            # Try to find raw JSON array
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response_content, re.DOTALL)
            if json_match:
                quiz_json = json_match.group(0)
            else:
                # Assume the entire response is JSON
                quiz_json = response_content
        
        # Parse the JSON
        quiz_data = json.loads(quiz_json)
        
        # Validate the data structure
        _validate_quiz_data(quiz_data)
        
        # Ensure we have the requested number of questions
        if len(quiz_data) > num_questions:
            quiz_data = quiz_data[:num_questions]
        
        # Add explanation field if missing
        for question in quiz_data:
            if "explanation" not in question:
                question["explanation"] = f"The correct answer is {question['correct_answer']}."
        
        return quiz_data
        
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error parsing quiz response: {str(e)}")
        
        # Create a fallback quiz if parsing fails
        return _create_fallback_quiz(subject, num_questions)
    



def generate_quiz(subject, level,topic, num_questions=5, reveal_answer=True):
    """
    Generate a quiz with multiple-choice questions based on subject and level.
    
    Args:
        subject (str): The academic subject
        level (str): Learning level (Beginner, Intermediate, Advanced)
        num_questions (int): Number of questions to generate
        reveal_answer (bool): Whether to format the response with hidden answers that can be revealed
    
    Returns:
        dict: Contains quiz data (list of questions) and formatted HTML if reveal_answer is True
    """
    try:
        num_questions = int(num_questions)
        # Get LLM instance
        llm = get_llm()
        
        # Create a structured prompt for quiz generation
        prompt = _create_quiz_prompt(subject, level,topic, num_questions)
        
        # Generate response
        logger.info(f"Generating quiz for subject: {subject} on topic {topic} , level: {level}, questions: {num_questions}")
        response = llm.invoke(prompt)
        
        # Parse and validate the response
        quiz_data = _parse_quiz_response(response.content, subject, num_questions)
        
        # Format the quiz with hidden answers if requested
        if reveal_answer:
            formatted_quiz = _format_quiz_with_reveal(quiz_data)
            return {
                "quiz_data": quiz_data,
                "formatted_quiz": formatted_quiz
            }
        else:
            return {
                "quiz_data": quiz_data
            }
        
    except Exception as e:
        logger.error(f"Error generating quiz: {str(e)}")
        raise Exception(f"Failed to generate quiz: {str(e)}")
    



def _format_quiz_with_reveal(quiz_data):
    """
    Format quiz data into HTML with hidden answers that can be revealed on click.
    
    Args:
        quiz_data (list): List of question dictionaries
        
    Returns:
        str: HTML string with quiz questions and hidden answers
    """
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: Arial, sans-serif;
                color: white;
                background-color: #121212;
            }
            .quiz-container {
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            .question {
                margin-bottom: 30px;
                padding: 20px;
                border: 1px solid #444;
                border-radius: 10px;
                background-color: #1e1e2f;
            }
            .question h3 {
                margin-top: 0;
                color: #90caf9;
            }
            .options {
                margin-left: 10px;
            }
            .option {
                margin: 10px 0;
                padding: 12px;
                border: 1px solid #555;
                border-radius: 6px;
                cursor: pointer;
                background-color: #2d2d44;
                transition: background-color 0.2s;
            }
            .option:hover {
                background-color: #3a3a5a;
            }
            .reveal-btn {
                background-color: #2196f3;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                font-weight: bold;
                margin-top: 15px;
                transition: background-color 0.2s;
            }
            .reveal-btn:hover {
                background-color: #0d8bf2;
            }
            .answer-section {
                margin-top: 20px;
                border: 2px solid #ffeb3b;
                border-radius: 8px;
                padding: 0;
                overflow: hidden;
                display: none;
            }
            .answer-header {
                background-color: #ffeb3b;
                color: #000;
                padding: 10px;
                font-weight: bold;
                font-size: 16px;
                text-align: center;
            }
            .answer-content {
                padding: 15px;
                background-color: #1a237e;
            }
            .correct-answer {
                font-size: 18px;
                font-weight: bold;
                color: white;
                margin-bottom: 15px;
            }
            .explanation {
                color: #e1f5fe;
                font-size: 16px;
                line-height: 1.5;
            }
            .selected-correct {
                background-color: #1b5e20 !important;
                border-color: #4caf50 !important;
            }
            .selected-incorrect {
                background-color: #b71c1c !important;
                border-color: #f44336 !important;
            }
        </style>
    </head>
    <body>
        <div class="quiz-container">
            <h2 style="color: #2196f3; text-align: center; margin-bottom: 30px;">Interactive Quiz</h2>
    """
    
    for i, question in enumerate(quiz_data, 1):
        option_letters = ["A", "B", "C", "D"]
        correct_index = question["options"].index(question["correct_answer"]) if question["correct_answer"] in question["options"] else 0
        
        html += f"""
            <div class="question" id="question-{i}">
                <h3>Question {i}</h3>
                <p>{question["question"]}</p>
                <div class="options">
        """
        
        for j, option in enumerate(question["options"]):
            is_correct = j == correct_index
            html += f"""
                    <div class="option" id="option-{i}-{j}" onclick="selectOption({i}, {j}, {is_correct})">
                        <strong>{option_letters[j]}.</strong> {option}
                    </div>
            """
        
        html += f"""
                </div>
                <button class="reveal-btn" onclick="revealAnswer({i})">SHOW ANSWER</button>
                <div class="answer-section" id="answer-{i}">
                    <div class="answer-header">CORRECT ANSWER</div>
                    <div class="answer-content">
                        <div class="correct-answer">{option_letters[correct_index]}. {question["correct_answer"]}</div>
                        <div class="explanation">{question.get("explanation", "")}</div>
                    </div>
                </div>
            </div>
        """
    
    html += """
        </div>
        <script>
            function selectOption(questionNum, optionNum, isCorrect) {
                const questionId = `question-${questionNum}`;
                const options = document.querySelectorAll(`#${questionId} .option`);
                
                // Reset all options
                options.forEach(option => {
                    option.className = 'option';
                });
                
                // Highlight selected option
                const selectedOption = document.getElementById(`option-${questionNum}-${optionNum}`);
                if (isCorrect) {
                    selectedOption.className = 'option selected-correct';
                } else {
                    selectedOption.className = 'option selected-incorrect';
                    // Show answer if incorrect
                    revealAnswer(questionNum);
                }
            }
            
            function revealAnswer(questionNum) {
                const answerDiv = document.getElementById(`answer-${questionNum}`);
                answerDiv.style.display = 'block';
                
                // Scroll to answer
                setTimeout(() => {
                    answerDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }, 100);
                
                // Add animation for attention
                answerDiv.animate([
                    { transform: 'scale(1)', boxShadow: '0 0 0 rgba(255, 235, 59, 0)' },
                    { transform: 'scale(1.03)', boxShadow: '0 0 20px rgba(255, 235, 59, 0.7)' },
                    { transform: 'scale(1)', boxShadow: '0 0 10px rgba(255, 235, 59, 0.3)' }
                ], {
                    duration: 1000,
                    iterations: 1
                });
            }
        </script>
    </body>
    </html>
    """
    
    return html

# Export quiz to file (new function)
def export_quiz_to_html(quiz_data, file_path="quiz.html"):
    """
    Export the formatted quiz to an HTML file
    
    Args:
        quiz_data (list): List of question dictionaries
        file_path (str): Path to save the HTML file
    """
    try:
        html_content = _format_quiz_with_reveal(quiz_data)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        logger.info(f"Quiz exported successfully to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error exporting quiz to HTML: {str(e)}")

import streamlit as st
from streamlit.components.v1 import html
import uuid
import os
from dotenv import load_dotenv

# 🔑 Load your environment variables (like GROQ_API_KEY if needed)
load_dotenv()



# ⚙️ Streamlit page config
st.set_page_config(page_title="📚 AI Tutor", layout="wide")
st.title("🎓 Quizify: AI Study Buddy")

# Sidebar for preferences
with st.sidebar:
    st.header("Customize Your Learning")
    subject = st.selectbox("📖 Select Subject",
                           ["Mathematics", "Physics", "Computer Science",
                            "History", "Biology", "Programming", "Business Studies",
                            "Accountancy", "Economics", "Reasoning", "English"])
    
    level = st.selectbox("📚 Select Learning Level",
                         ["Beginner", "Intermediate", "Advanced"])
    
    learning_style = st.selectbox("🧠 Pick a Learning Style",
                                  ["Visual", "Text-based", "Hands-on"])
    
    language = st.selectbox("🌍 Preferred Language",
                            ["English", "Hindi", "Spanish", "French"])
    
    background = st.selectbox("📊 Your Learning Stage",
                              ["Beginner", "Some Knowledge", "Experienced"])

# Tabs for Tutoring and Quiz
tab1, tab2 = st.tabs(["💬 Learn Something", "📝 Test Yourself"])

# 🔍 Tutor Tab
with tab1:
    st.header("💬 Ask an AI Tutor")
    question = st.text_area("💡 What topic or concept do you want to explore?",
                            "Explain Newton's Second Law of Motion.")

    if st.button("Get Explanation 🧠"):
        with st.spinner("📚 Crafting a clear explanation just for you..."):
            try:
                explanation = generate_tutoring_response(
                    subject=subject,
                    level=level,
                    question=question,
                    learning_style=learning_style,
                    background=background,
                    language=language
                )
                st.success("Here's your clear explanation:")
                st.markdown(explanation, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"⚠️ Error generating explanation: {str(e)}")

# 📝 Quiz Tab
with tab2:
    st.header("📘 Quick Quiz Time")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        num_questions = st.slider("Number of Questions", min_value=1, max_value=10, value=5)
        topic = st.text_input("Topic's Name")
    
    with col2:
        quiz_button = st.button("Generate Quiz 📝", use_container_width=True)
    
    if quiz_button and topic.strip():
        with st.spinner("Generating quiz questions..."):
            try:
                quiz_result = generate_quiz(
                    subject=subject,
                    level=level,
                    topic=topic,
                    num_questions=num_questions,
                    reveal_answer=True
                )
                
                st.success("Quiz generated! Try answering these questions:")

                if "formatted_quiz" in quiz_result and quiz_result["formatted_quiz"]:
                    html(quiz_result["formatted_quiz"],
                         height=max(500, num_questions * 350),
                         scrolling=True)

                    # Save for download
                    with open("generated_quiz.html", "w", encoding="utf-8") as f:
                        f.write(quiz_result["formatted_quiz"])

                    with open("generated_quiz.html", "rb") as file:
                        st.download_button(
                            label="📥 Download Quiz as HTML",
                            data=file,
                            file_name="quiz.html",
                            mime="text/html"
                        )
                else:
                    for i, q in enumerate(quiz_result["quiz_data"]):
                        with st.expander(f"Question {i+1}: {q['question']}", expanded=True):
                            session_id = str(uuid.uuid4())
                            selected = st.radio(
                                "Select your answer:",
                                q["options"],
                                key=f"q_{session_id}"
                            )
                            if st.button("Check Answer", key=f"check_{session_id}"):
                                if selected == q["correct_answer"]:
                                    st.success(f"✅ Correct! {q.get('explanation', '')}")
                                else:
                                    st.error(f"❌ Incorrect. The correct answer is: {q['correct_answer']}")
                                    if "explanation" in q:
                                        st.info(q["explanation"])
            except Exception as e:
                st.error(f"⚠️ Error generating quiz: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Powered by AI - Your Personal Learning Assistant")

