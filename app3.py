import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer # type: ignore    
from sklearn.metrics.pairwise import cosine_similarity
# LangChain model wrapper for Gemini
from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
# Still need the types for the safety settings values
from google.genai.types import HarmCategory, HarmBlockThreshold # type: ignore
import os # ESSENTIAL: Required to read the environment variable loaded by dotenv
from dotenv import load_dotenv  # type: ignore

# --- 1. LLM INITIALIZATION USING ENVIRONMENT VARIABLE (INCLUDING .ENV) ---

# Load environment variables from a potential .env file for local development
load_dotenv()


# Load key directly from environment variable (user must set GOOGLE_API_KEY externally)
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    st.error("Configuration Error: GOOGLE_API_KEY environment variable not set.")
    st.info("To run the app, please set your Gemini API key as the GOOGLE_API_KEY environment variable.")
    st.stop()
    
# Initialize the LangChain Chat Model (llm)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.7,
    model_kwargs={
        "config": { # Use 'config' dictionary as the dedicated place for non-standard parameters
            "max_output_tokens": 300, # Max tokens moved here
            "safety_settings": [
                {'category': HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, 'threshold': HarmBlockThreshold.BLOCK_NONE},
                {'category': HarmCategory.HARM_CATEGORY_HATE_SPEECH, 'threshold': HarmBlockThreshold.BLOCK_NONE},
                {'category': HarmCategory.HARM_CATEGORY_HARASSMENT, 'threshold': HarmBlockThreshold.BLOCK_NONE},
                {'category': HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, 'threshold': HarmBlockThreshold.BLOCK_NONE},
                #{'category': HarmCategory.HARM_CATEGORY_SELF_HARM, 'threshold': HarmBlockThreshold.BLOCK_NONE},
            ],
            "stop_sequences": ["###"]
        }
    }
)


st.set_page_config(page_title="Personalized Learning Path Generator",page_icon=":computer:", layout="wide")
st.title("🎯 Personalized Learning Path Generator")
st.write("Enter your current skills to get a personalized learning roadmap:")

# --- Data Loading and Embedding Functions ---
# NOTE: You must have a 'learnings.csv' file in the same directory for this to run.
@st.cache_data
def load_data():
    """Loads and preprocesses the course data from learnings.csv."""
    try:
        df = pd.read_csv("learnings.csv")
    except FileNotFoundError:
        st.error("Error: 'learnings.csv' not found. Please create this file with 'title', 'skills', and 'description' columns.")
        return pd.DataFrame({'title': [], 'skills': [], 'description': [], 'similarity': []})
        
    df['skills'] = df['skills'].apply(
        lambda x: [s.strip() for s in str(x).split(',') if s.strip() != '']
    )
    df['skills'] = df['skills'].apply(lambda x: ", ".join(x))
    return df

courses_df = load_data()

@st.cache_resource
def get_embeddings(descriptions):
    """Loads the SentenceTransformer model and computes embeddings."""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model.encode(descriptions)
    except Exception as e:
        # This will be caught by Streamlit's caching mechanism or the main app flow
        st.error(f"Error loading embedding model: {e}")
        return np.zeros((len(descriptions), 384))

course_embeddings = get_embeddings(courses_df['description'].tolist())

# --- Streamlit UI Elements ---
user_skills = st.text_input("Enter your skills (comma-separated):", "Python, Data Analysis")

field = st.selectbox("📘 Choose a Domain", ["Data Science", "Web Development", "AI/ML", "Finance", "Cybersecurity", "Blockchain", "Design"])
experience = st.selectbox("📊 Your Current Level", ["Beginner", "Intermediate", "Advanced"])
learning_style = st.selectbox("🧠 Preferred Style", ["Project-based", "Theory-focused", "Balanced"])
time_per_week = st.slider("🕒 Hours you can study per week", 1, 40, 8)

# --- Button Logic ---
if st.button("Generate Learning Path"):
    with st.spinner("Building your roadmap..."):
        if courses_df.empty:
             st.warning("Cannot generate path: Course data could not be loaded.")
        elif user_skills.strip() == "":
            st.warning("Please enter at least one skill.")
        else:
            user_skills = [skill.strip() for skill in user_skills.split(',') if skill.strip() != '']
            user_skills_text = ", ".join(user_skills)
            
            # Get user skills embedding
            user_embedding = get_embeddings([user_skills_text])
            
            # Compute similarity
            similars = cosine_similarity(user_embedding, course_embeddings)
            courses_df['similarity'] = similars[0]

            # Recommend top 5 courses
            recommended_courses = courses_df.sort_values(by='similarity', ascending=False).head(5)
            st.subheader("Top Recommended Courses:")
            st.dataframe(recommended_courses[['title', 'skills', 'similarity']])
            
            st.subheader("📚 Top Recommended Courses")
            top_courses = recommended_courses['title'].tolist()
            
            # Use the more detailed prompt for a better response
            prompt = f"""
            I am a {experience} learner interested in {field}. I prefer a {learning_style} learning style and can dedicate {time_per_week} hours per week to study. 
            Based on my current skills: {user_skills_text}, please create a personalized learning path using the following courses: {', '.join(top_courses)}.
            Structure the learning path with course titles and brief descriptions, ensuring a logical progression from foundational to advanced topics.
            """
            
            try:
                # Use the LangChain model's .invoke() method
                lc_response = llm.invoke(prompt)

                generated_text = lc_response.content

                print(generated_text)
                st.markdown("### 🛤️ Your Personalized Learning Path:")
                st.write(generated_text)
            
            except Exception as e:
                 # Note: LangChain exceptions may be nested, this catches the top layer
                 st.error(f"An API Error Occurred during generation: {e}")
                 st.info("Please ensure your API key is valid and the 'langchain-google-genai' package is installed.")
