# Importing dependencies
import os

from nltk import word_tokenize
import re
import pytube
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import NLTKTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.chains.question_answering import load_qa_chain
from youtube_transcript_api import YouTubeTranscriptApi
nltk.download('punkt')


def intro():
    st.write("# Welcome to the YouTube Toolkit 🛠️🎥! 👋🏻")
    st.sidebar.success("Select a Toolkit from above!")
    
    st.markdown(
    """
    Welcome to our innovative and unique YouTube Toolkit 🛠️🎥!

    Our web application has been exclusively designed to serve creators, viewers and everyone who interacts with YouTube content. The toolkit comes with three power-packed features:

    **Video Summary Generator** 📝🎥 - Do you have a video URL and want to know what it's about quickly? Simply pop in the URL and get a concise, informative summary of the video. Perfect for getting the gist of content in a pinch!

    **Interactive Chat with Video** 💬🎬 - Dive deeper into the content of a YouTube video. This unique feature allows you to interactively chat with the video content, extracting insights and gaining a richer understanding.

    **Script Writer** ✍️📜 - Do you have an amazing idea for a video but struggle to put it into words? Let our Script Writer assist you! Just input the topic and let the app generate a comprehensive script to kickstart your content creation journey.

    Give our toolkit a try and experience the difference it can bring to your YouTube journey. You've got the ideas, we've got the tools. Let's bring your YouTube vision to life! 💫
    **To get started, select a tool from the dropdown menu on the left** 👈. 
    Remember, the world of YouTube is just a click away!

    Happy creating! 🎉
    """
    )
    
    
apikey = st.sidebar.text_input("API-Key", type='password')
MODEL = st.sidebar.selectbox(label='Model', options=['Select a Model','gpt-3.5-turbo', 'text-davinci-003', 'text-davinci-002', 'text-ada-001','code-davinci-002'])
if apikey:
    os.environ['OPENAI_API_KEY'] = apikey

    
def get_video_id(video_url):
    video_id = video_url.split("v=")
    if len(video_id) > 1:
        return video_id[1]
    video_id = video_url.split("youtu.be/")
    if len(video_id) > 1:
        return video_id[1]
    
def get_video_transcript(video_id):
    try:
        script = YouTubeTranscriptApi.get_transcript(video_id)
        text_values = [obj['text'] for obj in script]
            
        transcript = ' '.join(text_values)
        transcript = transcript.replace("\xa0", " ")
        return (transcript)
    except:
        return ("Transcript is not available for this video")
    
def summarize_frags(fragment):
    llm = OpenAI(temperature=0, model_name=MODEL, max_tokens=2049)
    chain = load_summarize_chain(llm=llm, chain_type="map_reduce", verbose=True)
    docs = [Document(page_content=fragment)]
    return (chain.run(docs))

def splitting_text_to_chunks(text):
    splitter = NLTKTextSplitter(chunk_size=500)
    split_transcript = splitter.split_text(text)
    return (split_transcript)
    
def video_summary():
    st.markdown(f'# {list(page_names_to_funcs.keys())[1]}')
    st.markdown(
    """
    ### Pop in your YouTube URL and get a concise and informative summary of the whole video.
    """
    )
    video_url = st.text_input("Enter YouTube Video URL here", placeholder="Youtube URL...")
    if video_url:
        video_id = get_video_id(video_url)
        script = get_video_transcript(video_id)
        if script != "Transcript is not available for this video":
            summary = script
            summaryText = ""
            
            while len(word_tokenize(summary)) > 400:
                split_transcript = splitting_text_to_chunks(summary)
                
                for doc in split_transcript:
                    summaryText = summaryText + summarize_frags(doc)
                    
                summary = summaryText
                
            st.markdown("### Summary:")
            st.markdown(f'#### 📽️📹🎥: {pytube.YouTube(video_url).title}')
            st.write(summary)
        else:
            st.markdown("### Sorry could not process the request 🥺")
            st.write("Transcript is not available for this video! Can try providing us with another link for the same topic.")
        
        
        
def get_ques():
    """
    Get the user input question or query
    Returns:
        (str): The text entered by the user
    """
    question = st.text_input("What would you like to know from the video?", 
                             st.session_state["input"], key="input", placeholder="Enter your query here!")
    return question
        
def video_chat():
    st.markdown(f'# {list(page_names_to_funcs.keys())[2]}')
    st.markdown(
    """
    #### Pop in your YouTube URL and get a chance to interact with the video!
    """
    )
    video_url = st.text_input("Enter YouTube Video URL here", placeholder="Youtube URL...")
        
    if video_url:
        video_id = get_video_id(video_url)
        script = get_video_transcript(video_id)
        if script != "Transcript is not available for this video":
            split_transcript = splitting_text_to_chunks(script)
            
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(split_transcript, embedding=embeddings)
            
            if "generated" not in st.session_state:
                st.session_state["generated"] = []  # saves the output
            if "past" not in st.session_state:
                st.session_state["past"] = []   # saves the past
            if "input" not in st.session_state:
                st.session_state["input"] = ""  # saves the input
            if "stored_session" not in st.session_state:
                st.session_state["stored_session"] = []
            
            llm = OpenAI(temperature=0, model_name=MODEL)
            if "entity_memory" not in st.session_state:
                st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=10)
            
            user_input = get_ques()
            
            if user_input:
                st.markdown(f'#### 📽️📹🎥: {pytube.YouTube(video_url).title}')
                docs = VectorStore.similarity_search(query=user_input, k=3)
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question=user_input)
                st.session_state.past.append(user_input)
                st.session_state.generated.append(response)
                
            with st.expander("**Conversation** 📡"):
                for i in range(len(st.session_state['generated'])-1,-1,-1):
                    st.info(st.session_state["past"][i], icon="😃")
                    st.success(st.session_state["generated"][i], icon="🤖")
        
        else:
            st.markdown("#### Sorry could not process the request 🥺")
            st.write("Transcript is not available for this video! Can try providing us with another link for the same topic.")
                    
          
def script_writer():
    st.markdown(f'# {list(page_names_to_funcs.keys())[3]}')
    st.markdown(
    """
    #### Unleash your creativity with our Script Writer by letting us know your video idea and craft an engaging script for you!
    """
    )
    prompt = st.text_input("Plug in your prompt here ->", placeholder="Let us know the Topic...")
    # Prompt Templates
    title_template = PromptTemplate(
        input_variables = ['topic'],
        template = 'Suggest me a Youtube Video Title about {topic}'
    )

    script_template = PromptTemplate(
        input_variables = ['title', 'wikipedia_research'],
        template = 'Generate a Youtube Video Script based on the title: {title} while leveraging this wikipedia research: {wikipedia_research}'
    )
    
    # Memory
    title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
    script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')
    
    # Large Language Models
    llm = OpenAI(temperature=0.9, model_name=MODEL)
    title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
    script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)
    # sequential_chain = SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'], output_variables=['title', 'script'], verbose=True)
    wiki = WikipediaAPIWrapper()
    
    # Display's the output if there is a prompt
    if prompt:
        # response = sequential_chain({'topic':prompt})
        title = title_chain.run(topic=prompt)
        wiki_research = wiki.run(prompt)
        script = script_chain.run(title=title, wikipedia_research=wiki_research)
        
        st.write(title)
        st.write(script)
        
        with st.expander('Title History🔠⌛'):
            st.info(title_memory.buffer)
        
        with st.expander('Script History📜⌛'):
            st.info(script_memory.buffer)
            
        with st.expander('Wikipedia Research'):
            st.info(wiki_research)
    
    

page_names_to_funcs = {
    "Select a Toolkit": intro,
    "Video Summary Generator 📝🎥": video_summary,
    "Interactive Chat with the Video 💬🎬": video_chat,
    "Script Writer ✍️📜": script_writer
}

demo_name = st.sidebar.selectbox("Choose a Toolkit", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
