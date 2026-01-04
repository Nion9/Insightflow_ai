import streamlit as st
from processor import download_video_and_extract_audio, transcribe_audio, cleanup_temp_files
from brain import create_searchable_db
import os

# Page configuration
st.set_page_config(page_title="InsightFlow AI", page_icon="🚀", layout="wide")

# Title and description
st.title("🎥 InsightFlow: Media Intelligence")
st.markdown("Transform YouTube videos into searchable knowledge with AI")

# Initialize session state
if 'db' not in st.session_state:
    st.session_state['db'] = None
if 'transcript' not in st.session_state:
    st.session_state['transcript'] = None

# Input section
video_url = st.text_input("Paste YouTube URL here:", placeholder="https://www.youtube.com/watch?v=...")

# Process video button
if st.button("🚀 Analyze Video", type="primary"):
    if not video_url:
        st.error("Please enter a YouTube URL")
    else:
        try:
            with st.status("Processing... (This may take a minute)", expanded=True) as status:
                # Step 1: Download audio
                st.write("📥 Downloading audio...")
                audio_file = download_video_and_extract_audio(video_url)
                
                # Step 2: Transcribe
                st.write("🤖 Transcribing with AI...")
                transcript = transcribe_audio(audio_file)
                st.session_state['transcript'] = transcript
                
                # Step 3: Create vector database
                st.write("🧠 Building Knowledge Base...")
                db = create_searchable_db(transcript)
                st.session_state['db'] = db
                
                # Cleanup
                cleanup_temp_files()
                
                status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
            
            # Display transcript preview
            st.success("Processing complete! You can now ask questions about the video.")
            with st.expander("📄 View Full Transcript", expanded=False):
                st.text_area(
                    "Transcript:", 
                    st.session_state['transcript'], 
                    height=300,
                    label_visibility="collapsed"
                )
            
        except Exception as e:
            st.error(f"❌ Error processing video: {str(e)}")
            cleanup_temp_files()

# Divider
if st.session_state['db'] is not None:
    st.divider()
    
    # Chat section
    st.subheader("💬 Ask Questions About the Video")
    
    # Question input
    query = st.text_input(
        "Your question:", 
        placeholder="What is this video about?",
        key="question_input"
    )
    
    if query:
        try:
            with st.spinner("Searching knowledge base..."):
                # Search the database for relevant content
                docs = st.session_state['db'].similarity_search(query, k=3)
                
                if docs:
                    st.markdown("**📌 Top Insights:**")
                    
                    # Display top result prominently
                    st.info(f"**Most Relevant:**\n\n{docs[0].page_content}")
                    
                    # Show additional results in expander
                    if len(docs) > 1:
                        with st.expander("🔍 See more relevant sections"):
                            for i, doc in enumerate(docs[1:], 2):
                                st.markdown(f"**Result {i}:**")
                                st.write(doc.page_content)
                                st.divider()
                else:
                    st.warning("No relevant information found. Try rephrasing your question.")
                    
        except Exception as e:
            st.error(f"Error searching database: {str(e)}")

# Sidebar with instructions
with st.sidebar:
    st.header("ℹ️ How to Use")
    st.markdown("""
    1. **Paste** a YouTube URL
    2. **Click** "Analyze Video"
    3. **Wait** for processing (1-3 minutes)
    4. **Ask** questions about the content
    
    ---
    
    ### Features:
    - 🎯 AI-powered transcription
    - 🧠 Semantic search
    - 💬 Interactive Q&A
    - 📊 Full transcript access
    
    ---
    
    ### Tips:
    - Works best with English videos
    - Longer videos take more time
    - Ask specific questions for best results
    """)
    
    # Add a reset button
    if st.button("🔄 Reset Session"):
        st.session_state['db'] = None
        st.session_state['transcript'] = None
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Built with Streamlit, Whisper & ChromaDB</div>",
    unsafe_allow_html=True
)