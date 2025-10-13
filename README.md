# Task3
🏥 Medical Q&A Chatbot
This project develops a specialized medical question-answering chatbot designed to provide information based on the MedQuAD dataset. The chatbot uses a retrieval-based approach to find relevant answers and incorporates medical entity recognition to understand user queries more deeply. The user interface is built with Streamlit, making it accessible and easy to use.

1. Problem Statement
The rapid expansion of medical information presents a significant challenge for individuals seeking reliable health answers. Traditional search engines can provide vast, but often disorganized, results that fail to interpret complex medical terms or provide contextually relevant answers. The goal of this project is to address this by creating a reliable, accessible, and specialized medical Q&A system. It should accurately retrieve information from a trusted medical source, recognize key entities like diseases and symptoms, and present the information in a clear, user-friendly format.

2. Dataset
The chatbot's knowledge base is built upon the MedQuAD (Medical Question Answering Dataset).

Content: MedQuAD is a collection of over 47,457 question-answer pairs curated from 12 trusted National Institutes of Health (NIH) websites, including MedlinePlus and cancer.gov. It covers a wide range of health topics and includes 37 distinct question types, such as treatment options, diagnosis, and side effects.

Preprocessing (data_processor.py): The data is loaded from Hugging Face and undergoes a rigorous cleaning process. This involves removing empty or short questions and answers, dropping duplicates, and normalizing text. The processor also classifies questions into categories (e.g., 'symptoms', 'treatment', 'diagnosis') and adds metadata like complexity scores. This ensures the knowledge base is clean, structured, and ready for efficient retrieval.

3. Methodology
The chatbot's architecture is a retrieval-based system, which combines a vector database with medical-specific NLP techniques to answer user queries.

Data Processing & Indexing (data_processor.py): The preprocessed MedQuAD dataset is used to build a search index. Using the sentence-transformers library, each question and its corresponding answer from the dataset are transformed into dense vector embeddings. These embeddings are then stored in a FAISS index for fast and efficient similarity search.

Medical Entity Recognition (entity_recognition.py): Before the search, the user's query is processed by a Medical Entity Recognizer. This component uses both the spaCy library (with a medical-specific model like en_core_sci_sm) and a set of custom regex patterns to identify key medical entities like diseases, symptoms, and treatments.

Information Retrieval: When a user asks a question, the chatbot converts the query into an embedding and performs a semantic search on the FAISS index to find the most relevant question-answer pairs. The system retrieves the top k most similar documents based on their similarity score.

User Interface (app (3).py): A simple, yet effective, user interface is built with Streamlit. It allows users to input medical questions, see the retrieved answers, and view a dashboard with real-time performance metrics and entity analysis.

4. Results
The project successfully delivers a specialized medical Q&A chatbot that is both accurate and user-friendly.

Accurate Retrieval: By leveraging the MedQuAD dataset and semantic search, the chatbot provides highly relevant and accurate medical information for domain-specific queries. The system is able to achieve high precision and recall, as demonstrated by the evaluation metrics.

Entity-Aware Responses: The medical entity recognition feature allows the chatbot to identify and highlight key terms in the user's query and the corresponding answer, providing a more structured and informative user experience.

Comprehensive Evaluation: The project includes a dedicated evaluation module (evaluation.py) that tracks key performance metrics such as precision, recall, and F1-score. This provides a quantitative measure of the system's effectiveness and helps to identify areas for improvement.

Reproducibility: The project is designed to be easily reproducible, with all dependencies listed in requirements (3).txt and the data loaded directly from a trusted source.
