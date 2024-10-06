# imports for langchain, streamlit
import streamlit as st

# This must be the first Streamlit command used on an app page, 
# and must only be set once per page.
st.set_page_config(
    page_title='Finance News Chatbot',
    page_icon='💰',
#     initial_sidebar_state='collapsed'
)

# Use vector store and chat history as context to answer user's questions
def ask_and_get_answer(vector_store, query, chat_history, k=3):
    from langchain_openai import ChatOpenAI
    from langchain.chains.history_aware_retriever import create_history_aware_retriever
    from langchain.chains.retrieval import create_retrieval_chain
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.chains.combine_documents import create_stuff_documents_chain

    # print(chat_history)

    # Instantiate LLM
    llm = ChatOpenAI(
        model='gpt-4o-mini', 
        temperature=1)
    
    # Instantiate vector store retriever
    retriever = vector_store.as_retriever(
        search_type='similarity', 
        search_kwargs={'k': k})

    # Create a history aware retreiver that takes into account the conversation history
    condense_question_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation and a follow up question, \
         rephrase the follow up question to be a standalone question.")
    ])

    history_retriever_chain = create_history_aware_retriever(
        llm, 
        retriever, 
        condense_question_prompt
        )
    
    # Create a document chain prompt to send to the LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Use the following pieces of context to answer the user's question. \
        If you don't know the answer, just say that you don't know, \
        don't try to make up an answer.\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    # Create a chain for passing a list of Documents to a model
    document_chain = create_stuff_documents_chain(llm, prompt)

    # combine history aware retriever and document chain
    conversational_retrieval_chain = create_retrieval_chain(
        history_retriever_chain, 
        document_chain
        )
    
    # get response from the LLM
    response = conversational_retrieval_chain.invoke({
        "input": query,
        "chat_history": chat_history}
        )
    return response["answer"]

# Streamed response emulator
def stream_response_generator(answer):
    import time
    for word in answer.split():
        yield word + " "
        time.sleep(0.05)

# clear the chat history from streamlit session state
def clear_history():
    if 'messages' in st.session_state:
        del st.session_state['messages']

# Right justification for chatbot messages
st.html(
"""
<style>
    .stChatMessage:has(.chat-assistant) {
        flex-direction: row-reverse;
        text-align: right;
    }
</style>
"""
)

if __name__ == "__main__":
    import ast
    from pinecone import Pinecone
    from langchain_pinecone import PineconeVectorStore
    from langchain_openai import OpenAIEmbeddings
    import constants
    from datetime import datetime, timezone, timedelta

    # loading the OpenAI api key from .env
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.html(f"<span class='chat-{message['role']}'></span>")
            st.markdown(message["content"])
    
    # Get Pinecone vector store
    pc = Pinecone()
    index = pc.Index(host=constants.PINECONE_INDEX_HOST)
    embeddings = OpenAIEmbeddings(
        model=constants.OPENAI_EMBEDDING_MODEL,
        dimensions=constants.EMBEDDING_DIMENSIONS
    )
    vector_store = PineconeVectorStore(
        index=index, 
        embedding=embeddings,
        text_key="content"
    )
    # saving the vector store in the streamlit session state (to be persistent between reruns)
    st.session_state.vs = vector_store

    with st.sidebar:
        ## text_input for the OpenAI API key (alternative to python-dotenv and .env)
        ## api_key = st.text_input('OpenAI API Key:', type='password')
        ## if api_key:
        ##     os.environ['OPENAI_API_KEY'] = api_key

        # add data button widget
        clr_hist = st.button('Clear History', on_click=clear_history)

        # get Pinecone index stats
        st.markdown(index.describe_index_stats())

        # get date time for right now 
        rightnow = datetime.now(timezone.utc)
        epoch_rightnow = int(rightnow.timestamp())
        # print(f"UTC time={rightnow}, epoch={epoch_rightnow}")

        # calculate date time for 1 day ago
        one_day_ago = rightnow - timedelta(days=1)
        epoch_one_day_ago = int(one_day_ago.timestamp())
        # print(f"UTC time from 1 day ago={one_day_ago}, epoch={epoch_one_day_ago}")

        # get documents from the past 1 day
        docs_one_ago = vector_store.similarity_search(  
                        query="",  # the search query - empty to filter by date
                        k=100,  # return 100 most relevant docs
                        filter={    # filter for documents
                            "epoch": {"$gt": epoch_one_day_ago}
                        },
                        namespace="" # namespace to search in
        )
        # get ticker symbols for the past day from the metadata
        tickers = []
        for doc in docs_one_ago:
            # print(doc)
            # print(f"date={doc.metadata['date']}")
            tickers.extend(ast.literal_eval(doc.metadata['symbols']))
        st.markdown(f"I have news for these companies from the past day")
        st.markdown(f"{tickers}")

    # React to user input
    if prompt := st.chat_input("Ask a question about Finance news"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # if there's a vector store (with embedded financial news data)
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs

            with st.spinner('Working on your request ...'):
                # creating the LLM response
                answer = ask_and_get_answer(
                    vector_store, 
                    prompt,
                    st.session_state.messages)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.html(f"<span class='chat-assistant'></span>")
                response = st.write_stream(stream_response_generator(answer))
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

# run the app: streamlit run ./streamlit_app.py