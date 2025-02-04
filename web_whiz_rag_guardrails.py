import os
import gradio as gr
import nest_asyncio
import asyncio
from openai import OpenAI
from langchain_openai import  ChatOpenAI
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import numpy as np
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain import PromptTemplate
from sentence_transformers import SentenceTransformer, util
from nemoguardrails import RailsConfig, LLMRails
from langchain_core.output_parsers import StrOutputParser
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from langchain_core.runnables import  RunnablePassthrough


nest_asyncio.apply()
MODEL = "gpt-4o-mini"
db_name = "vector_db"
hf_db_name = "chroma_data"



override=True
openai_api_key = os.getenv('OPENAI_API_KEY')
hf_token = os.getenv('HF_TOKEN')
openai = OpenAI()
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
hf_vectorstore = Chroma(collection_name="webpage_content",
        embedding_function=hf_embeddings,
        persist_directory=hf_db_name)
print(f"HF Vectorstore has {hf_vectorstore._collection.count()} documents")

# create a new Chat with OpenAI
llm = ChatOpenAI(temperature=0.7, model_name=MODEL)

# set up the conversation memory for the chat
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# the retriever is an abstraction over the VectorStore that will be used during RAG
retriever = hf_vectorstore.as_retriever(search_kwargs={"k": 25})



# Define the prompt template


custom_prompt = PromptTemplate(
    template=(
        "You are an AI assistant that interacts with users in a friendly and professional manner. \
        Provide a helpful answer based on the provided context. \
        If the question is outside the scope of the provided context, respond politely \
        Chat_History: {chat_history} \
        Context: {context} \
        Question: {question} \
        Answer: \
       "
    ),
    input_variables=["chat_history","context","question"]
) 

def evaluate_similarity(query, answer, threshold=0.5):
    # Initialize embedding model
    embedding_model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    answer_embedding = embedding_model.encode(answer, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(query_embedding, answer_embedding).item()
    
    if similarity < threshold:
        print(f"Misaligned response detected! Similarity: {similarity:.2f}")
        return similarity
    else:
        print(f"Response aligned. Similarity: {similarity:.2f}")
        return similarity

class LoggingConversationalRetrievalChain(ConversationalRetrievalChain):
    def run(self, inputs):
        print("Running LoggingConversationalRetrievalChain")
        query = inputs["question"]
        answer = super().run(inputs)
        # Log or store query and answer
        print(f"Query: {query}")
        #print(f"Context: {answer['context']}")
        print(f"Answer: {answer}")
        similarity=evaluate_similarity(query, answer)
        print(f"Evaluating similarity: {similarity}")
        return answer

# putting it together: set up the conversation chain with the GPT 3.5 LLM, the vector store and memory
#conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
#conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory,combine_docs_chain_kwargs={"prompt": custom_prompt})
#conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, callbacks=[StdOutCallbackHandler()] ,combine_docs_chain_kwargs={"prompt": PROMPT})
custom_chain = LoggingConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory,combine_docs_chain_kwargs={"prompt": custom_prompt} )




config = RailsConfig.from_path("config")

guard_rails = RunnableRails(config=config)

"""rails=LLMRails(config)

response = rails.generate(messages=[{
    "role": "user",
    "content": "Hello! What can you do for me?"
}])
print(response["content"])

info = rails.explain()
info.print_llm_calls_summary()

print(info.llm_calls[0].prompt)"""



chain_guardrails=guard_rails | custom_chain



query = "what is Pydantic?"

rails_result = guard_rails.invoke( {"input" :query})
result = custom_chain.invoke( {"question" :query})
result = chain_guardrails.invoke( {"question" :query})
print(result)


"""
def chat(message, history):
    result = custom_chain.invoke({"question": message})
    return result["answer"]

view = gr.ChatInterface(chat).launch(inbrowser=True)
"""