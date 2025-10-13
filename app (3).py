import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_processor import MedQuADProcessor
from retrieval import MedicalRetriever
from entity_recognition import MedicalEntityRecognizer
from evaluation import EvaluationMetrics
import os
import time

# Page configuration
st.set_page_config(
    page_title="Medical Q&A Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .question-box {
        background-color: #fff9e6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffa500;
        margin: 0.5rem 0;
    }
    .entity-box {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_data():
    """Load and cache the dataset"""
    processor = MedQuADProcessor()
    
    # Check in the data directory
    data_file_path = os.path.join(processor.data_dir, "processed_medquad.pkl")
    
    if os.path.exists(data_file_path):
        df = processor.load_processed_data()
        st.success("✅ Loaded processed MedQuAD data")
    else:
        with st.spinner("🔄 Loading MedQuAD dataset from Hugging Face... This may take a few minutes."):
            df = processor.load_all_data()
            processor.save_processed_data(df)
        st.success("✅ Successfully loaded MedQuAD dataset!")
    
    return df

@st.cache_resource
def load_retriever(_df):
    """Load and cache the retriever"""
    retriever = MedicalRetriever()
    with st.spinner("🔨 Building search index... This may take a few minutes on first run."):
        retriever.build_index(_df)
    return retriever

@st.cache_resource
def load_entity_recognizer():
    """Load and cache the entity recognizer"""
    return MedicalEntityRecognizer()

@st.cache_resource
def load_evaluator(_df, _retriever):
    """Load and cache the evaluator"""
    return EvaluationMetrics(_df, _retriever)

def display_system_metrics(retriever, evaluator):
    """Display system performance metrics"""
    st.header("📊 System Performance")
    
    stats = retriever.get_statistics()
    eval_metrics = evaluator.compute_basic_metrics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Q&A Pairs", stats.get('total_qa_pairs', 0))
    with col2:
        st.metric("Question Types", stats.get('question_types', 0))
    with col3:
        st.metric("Avg Response Time", f"{eval_metrics.get('avg_response_time', 0):.2f}s")
    with col4:
        search_method = "Semantic" if stats.get('using_semantic_search', False) else "Keyword"
        st.metric("Search Method", search_method)
    
    # Display retrieval performance
    st.subheader("Retrieval Performance")
    
    try:
        perf_data = evaluator.evaluate_retrieval_performance()
        
        if perf_data:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Precision@3", f"{perf_data.get('precision', 0):.3f}")
            with col2:
                st.metric("Recall@3", f"{perf_data.get('recall', 0):.3f}")
            with col3:
                st.metric("F1-Score", f"{perf_data.get('f1_score', 0):.3f}")
    except Exception as e:
        st.info("Evaluation metrics will be available after running queries.")

def display_entity_analysis(entity_recognizer, df):
    """Display medical entity analysis"""
    st.header("🏥 Medical Entity Analysis")
    
    # Sample analysis on the dataset
    sample_size = min(50, len(df))
    sample_questions = df['question'].sample(sample_size).tolist()
    all_entities = []
    
    with st.spinner("Analyzing entities..."):
        for question in sample_questions:
            entities = entity_recognizer.extract_entities(question)
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    all_entities.append({'type': entity_type, 'entity': entity})
    
    if all_entities:
        entities_df = pd.DataFrame(all_entities)
        
        # Entity type distribution
        entity_counts = entities_df['type'].value_counts().reset_index()
        entity_counts.columns = ['Entity Type', 'Count']
        
        fig = px.bar(entity_counts, x='Entity Type', y='Count', 
                     title='Medical Entity Type Distribution',
                     color='Count',
                     color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
        
        # Top entities
        top_entities = entities_df['entity'].value_counts().head(10).reset_index()
        top_entities.columns = ['Entity', 'Count']
        
        fig2 = px.pie(top_entities, values='Count', names='Entity',
                     title='Top 10 Medical Entities')
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No entities detected in the sample.")

def main():
    # Header
    st.markdown('<p class="main-header">🏥 Advanced Medical Q&A Chatbot</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by MedQuAD Dataset | Enhanced with Evaluation Metrics</p>', unsafe_allow_html=True)
    
    # Medical disclaimer
    st.markdown("""
        <div class="warning-box">
        ⚠️ <strong>Medical Disclaimer:</strong> This chatbot is for informational purposes only. 
        It is not a substitute for professional medical advice, diagnosis, or treatment. 
        Always seek the advice of your physician or other qualified health provider.
        </div>
    """, unsafe_allow_html=True)
    
    # Load components
    try:
        df = load_data()
        retriever = load_retriever(df)
        entity_recognizer = load_entity_recognizer()
        evaluator = load_evaluator(df, retriever)
    except Exception as e:
        st.error(f"❌ Error loading components: {str(e)}")
        st.info("Please check your requirements.txt and ensure all dependencies are installed.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Information")
        display_system_metrics(retriever, evaluator)
        
        st.markdown("---")
        st.header("⚙️ Settings")
        
        retrieval_method = st.selectbox(
            "Retrieval Method",
            ["Semantic Search", "Hybrid Search", "Keyword Search"],
            index=0
        )
        
        top_k = st.slider("Number of results", min_value=1, max_value=10, value=3)
        similarity_threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        show_entities = st.checkbox("Show Medical Entities", value=True)
        show_similarity = st.checkbox("Show Similarity Scores", value=True)
        enable_reranking = st.checkbox("Enable Re-ranking", value=True)
        
        st.markdown("---")
        st.header("📚 Sample Questions")
        
        sample_questions = [
            "What are the symptoms of diabetes?",
            "How is hypertension treated?",
            "What causes asthma?",
            "What are the side effects of chemotherapy?",
            "How is pneumonia diagnosed?",
            "What is the treatment for migraine?",
            "What are the risk factors for heart disease?"
        ]
        
        for sq in sample_questions:
            if st.button(sq, key=sq):
                st.session_state.sample_query = sq
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat Interface", "📈 Analytics", "🔍 System Evaluation"])
    
    with tab1:
        chat_interface(retriever, entity_recognizer, evaluator, top_k, similarity_threshold, 
                      show_entities, show_similarity, enable_reranking, retrieval_method)
    
    with tab2:
        display_entity_analysis(entity_recognizer, df)
        display_retrieval_analytics(evaluator)
    
    with tab3:
        display_system_evaluation(evaluator)

def chat_interface(retriever, entity_recognizer, evaluator, top_k, similarity_threshold, 
                  show_entities, show_similarity, enable_reranking, retrieval_method):
    """Main chat interface"""
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'sample_query' not in st.session_state:
        st.session_state.sample_query = None
    if 'query_times' not in st.session_state:
        st.session_state.query_times = []
    
    st.header("💬 Ask Your Medical Question")
    
    # Input form
    with st.form(key="question_form", clear_on_submit=True):
        default_query = st.session_state.sample_query if st.session_state.sample_query else ""
        
        user_question = st.text_area(
            "Type your question here:",
            value=default_query,
            height=100,
            placeholder="e.g., What are the symptoms of diabetes?"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            submit_button = st.form_submit_button("🔍 Search", use_container_width=True)
        with col2:
            advanced_options = st.checkbox("Advanced")
        
        if st.session_state.sample_query:
            st.session_state.sample_query = None
    
    if submit_button and user_question:
        start_time = time.time()
        
        # Add user question to messages
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        # Extract entities from question
        question_entities = None
        if show_entities:
            try:
                with st.spinner("🔍 Extracting medical entities..."):
                    question_entities = entity_recognizer.extract_entities(user_question)
            except Exception as e:
                st.warning(f"Could not extract entities: {str(e)}")
        
        # Retrieve answers based on selected method
        results = []
        try:
            with st.spinner("📚 Searching for relevant answers..."):
                if retrieval_method == "Semantic Search":
                    results = retriever.semantic_search(user_question, top_k=top_k, 
                                                      similarity_threshold=similarity_threshold)
                elif retrieval_method == "Keyword Search":
                    results = retriever.keyword_search(user_question, top_k=top_k)
                else:  # Hybrid
                    results = retriever.hybrid_search(user_question, top_k=top_k)
                
                # Apply re-ranking if enabled
                if enable_reranking and results:
                    results = retriever.rerank_results(user_question, results)
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            results = []
        
        response_time = time.time() - start_time
        st.session_state.query_times.append(response_time)
        
        # Log query for evaluation
        try:
            evaluator.log_query(user_question, results, response_time)
        except Exception as e:
            pass  # Silent fail for logging
        
        # Store results in session
        st.session_state.messages.append({
            "role": "assistant",
            "content": results,
            "entities": question_entities,
            "response_time": response_time,
            "retrieval_method": retrieval_method
        })
    
    # Display conversation
    display_conversation(show_entities, show_similarity, entity_recognizer)

def display_conversation(show_entities, show_similarity, entity_recognizer):
    """Display the conversation history"""
    if st.session_state.messages:
        st.markdown("---")
        st.header("💡 Results")
        
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="question-box"><strong>Your Question:</strong> {message["content"]}</div>', 
                           unsafe_allow_html=True)
                
            else:
                results = message["content"]
                entities = message.get("entities")
                response_time = message.get("response_time", 0)
                retrieval_method = message.get("retrieval_method", "Semantic Search")
                
                # Display query info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"⏱️ Response Time: {response_time:.2f}s")
                with col2:
                    st.caption(f"🔧 Method: {retrieval_method}")
                with col3:
                    if results:
                        st.caption(f"📊 Results: {len(results)}")
                
                # Show extracted entities
                if show_entities and entities and any(entities.values()):
                    st.markdown('<div class="entity-box">', unsafe_allow_html=True)
                    st.markdown("**🔍 Detected Medical Entities:**")
                    entity_summary = entity_recognizer.get_entity_summary(entities)
                    st.markdown(entity_summary)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Show results
                display_results(results, show_similarity, entity_recognizer, show_entities)
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.session_state.query_times = []
            st.rerun()

def display_results(results, show_similarity, entity_recognizer, show_entities):
    """Display search results"""
    if not results:
        st.warning("No relevant results found. Try rephrasing your question or adjusting the similarity threshold.")
        return
    
    for i, result in enumerate(results, 1):
        question_preview = result['question'][:100] + ("..." if len(result['question']) > 100 else "")
        
        with st.expander(f"📋 Result {i}: {question_preview}", expanded=(i==1)):
            if show_similarity:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Similar Question:** {result['question']}")
                with col2:
                    similarity_percent = result['similarity'] * 100
                    st.metric("Match", f"{similarity_percent:.1f}%")
            else:
                st.markdown(f"**Similar Question:** {result['question']}")
            
            # Display answer with proper formatting
            st.markdown('<div class="answer-box">', unsafe_allow_html=True)
            st.markdown(f"**Answer:**")
            st.write(result["answer"])
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"📂 Type: {result.get('doc_type', 'N/A')}")
            with col2:
                st.caption(f"❓ Category: {result.get('question_type', 'N/A')}")
            with col3:
                st.caption(f"📚 Source: {result.get('source', 'MedQuAD')}")
            
            # Extract entities from answer
            if show_entities:
                try:
                    with st.spinner("Analyzing answer entities..."):
                        answer_entities = entity_recognizer.extract_entities(result['answer'])
                    if any(answer_entities.values()):
                        st.markdown("**Medical Entities in Answer:**")
                        st.markdown(entity_recognizer.get_entity_summary(answer_entities))
                except Exception as e:
                    pass  # Silent fail for entity extraction

def display_retrieval_analytics(evaluator):
    """Display retrieval analytics"""
    st.header("📈 Retrieval Analytics")
    
    # Performance metrics
    metrics = evaluator.compute_basic_metrics()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Queries", metrics.get('total_queries', 0))
    with col2:
        st.metric("Avg Results/Query", f"{metrics.get('avg_results_per_query', 0):.1f}")
    with col3:
        st.metric("Success Rate", f"{metrics.get('success_rate', 0):.1f}%")
    with col4:
        st.metric("Avg Similarity", f"{metrics.get('avg_similarity', 0):.3f}")
    
    # Response time distribution
    query_times = st.session_state.get('query_times', [])
    if len(query_times) > 1:
        fig = px.histogram(x=query_times, 
                          title="Response Time Distribution",
                          labels={'x': 'Response Time (seconds)', 'y': 'Frequency'},
                          nbins=20)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run more queries to see response time analytics.")

def display_system_evaluation(evaluator):
    """Display comprehensive system evaluation"""
    st.header("🔍 System Evaluation")
    
    with st.spinner("Running evaluation..."):
        try:
            evaluation_results = evaluator.comprehensive_evaluation()
        except Exception as e:
            st.error(f"Error running evaluation: {str(e)}")
            return
    
    if evaluation_results:
        # Overall metrics
        st.subheader("Overall Performance")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Reciprocal Rank", f"{evaluation_results.get('mrr', 0):.3f}")
        with col2:
            st.metric("Mean Average Precision", f"{evaluation_results.get('map', 0):.3f}")
        with col3:
            st.metric("NDCG@3", f"{evaluation_results.get('ndcg', 0):.3f}")
        with col4:
            st.metric("Precision@3", f"{evaluation_results.get('precision', 0):.3f}")
        
        # Performance by question type
        st.subheader("Performance by Question Type")
        perf_by_type = evaluation_results.get('performance_by_type', [])
        
        if perf_by_type:
            perf_df = pd.DataFrame(perf_by_type)
            fig = px.bar(perf_df, x='question_type', y='precision', 
                        title='Precision by Question Type',
                        color='precision',
                        color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        
        # Error analysis
        st.subheader("Error Analysis")
        error_analysis = evaluation_results.get('error_analysis', {})
        
        if error_analysis:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Common Failure Patterns:**")
                for pattern in error_analysis.get('common_failures', []):
                    st.write(f"- {pattern}")
            with col2:
                st.write("**Recommendations:**")
                for rec in error_analysis.get('recommendations', []):
                    st.write(f"- {rec}")

if __name__ == "__main__":
    main()