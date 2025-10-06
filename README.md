# Personalized Learning Path Generator

**Tech Stack:** Streamlit | LangChain | Google Gemini LLMs | Python | Sentence Transformers | AI


## **Executive Summary**

The Personalized Learning Path Generator is an **AI-powered web application** designed to create **customized learning roadmaps** for users based on their current skills, experience, preferred learning style, and domain. By leveraging Google Gemini LLMs and advanced NLP techniques, the app delivers **adaptive, structured, and actionable learning recommendations** from curated course datasets.

## **Business Problem**

Upskilling and continuous learning have become crucial in today’s rapidly evolving job market. Individuals often struggle to identify:

- Which courses align with their current skills
- How to progress from beginner to advanced levels efficiently
- How to structure their learning path logically

This project addresses these challenges by providing **personalized, AI-driven guidance**, saving learners time and improving learning efficiency.

## **Methodology**

1. **User Input:** Users provide their skills, experience level, domain, and learning preferences.
2. **Course Dataset:** Curated dataset containing course titles, skills covered, and descriptions.
3. **Skill Matching:** User input is converted into embeddings using **Sentence Transformers**, and similarity with course descriptions is calculated via **cosine similarity**.
4. **LLM Integration:** Google Gemini-2.5-flash LLM (via LangChain) generates **personalized, structured learning paths** with logical progression.
5. **UI & Interactivity:** **Streamlit** dashboard allows users to input data and receive real-time, readable learning roadmaps.

## **Learning & Skills Demonstrated**

- Large Language Models (LLMs) integration with LangChain
- Prompt engineering for AI content generation
- Data processing and embeddings with Python (Pandas, Sentence Transformers)
- Interactive dashboard development with Streamlit
- API key management and cloud-based AI model integration

## **Results**

- Delivered **adaptive learning paths** tailored to individual skills and goals, improving learning efficiency and engagement.
- Recommended courses in a **logical progression** from foundational to advanced topics, simplifying complex skill-to-course mapping.
- Enabled users to **quickly identify relevant courses**, making career planning and upskilling actionable and personalized.

## **Next Steps**

- Add **user authentication and profile management** to save learning paths.
- Enhance the LLM prompt logic for **personalized learning tips and timelines**.
- Implement **real-time feedback and course completion suggestions** using AI.
- Optimize the model for **scalability and faster response times** for multiple users.
