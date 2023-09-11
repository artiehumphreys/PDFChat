from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores import Qdrant
from langchain.document_loaders import PyPDFLoader
from langchain.llms import GPT4All
import qdrant_client
from langchain import PromptTemplate, LLMChain
import os
from timeit import default_timer as timer
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain import HuggingFacePipeline
#from transformers import AutoTokenizer, pipeline
import torch
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain import LLMChain
# embedding model options:
MPNET_MODEL = "sentence-transformers/all-mpnet-base-v2"
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MULTIQA_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

# Specify the embedding model here
model_name = MULTIQA_MODEL

# Specify the database option here ("faiss", "qdrant", "milvus")
db_option = "faiss"

# Create the HuggingFaceEmbeddings object
embeddings = HuggingFaceEmbeddings(model_name=model_name)

callback_manager = AsyncCallbackManager([StreamingStdOutCallbackHandler()])

#Specify the path to your LLM model here
gpt4all_path = '/models/ggml-gpt4all-j-v1.3-groovy.bin' 
DUSTIN_MODEL_PATH = "/models/GPT4All-13B-snoozy.ggmlv3.q8_0.bin"
#For HuggingFaceHub Models
repo_id = "tiiuae/falcon-7b-instruct"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"repetition_penalty": .5, "temperature": .5, "max_new_tokens": 1000})

#For Falcon 7B
Falcon_model = "tiiuae/falcon-7b-instruct" #tiiuae/falcon-40b-instruct

#Falcon_tokenizer = AutoTokenizer.from_pretrained(Falcon_model)

#pipeline = pipeline(
#    "text-generation", #task
#    model=Falcon_model,
#    tokenizer=Falcon_tokenizer,
#    torch_dtype=torch.bfloat16,
#    trust_remote_code=True,
#    device_map="auto",
#    max_length=200,
#    do_sample=True,
#    top_k=10,
#    num_return_sequences=1,
#    eos_token_id=Falcon_tokenizer.eos_token_id
#)
#llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

#Begin counting the total time
total_start = timer()

#Define LLM here. It will be different if using different LLM models
#llm = GPT4All(model=DUSTIN_MODEL_PATH, callback_manager=callback_manager, verbose=True)

# Split text 
def split_chunks(sources):
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=32)
    for chunk in splitter.split_documents(sources):
        chunks.append(chunk)
    return chunks


def create_index(chunks, db_option):
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    if db_option == "faiss" or db_option == "FAISS":
        search_index = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    elif db_option == "qdrant" or db_option == "Qdrant" or db_option == "QDRANT":
        search_index = Qdrant.from_texts(texts, embeddings, metadatas=metadatas, location=":memory:", collection_name="my_qdrant_index1")
        #search_index = Qdrant.from_texts(texts, embedding=embeddings, location=":memory:")
        #client = qdrant_client.QdrantClient(":memory:")
        #search_index = Qdrant.from_texts(client=client, collection_name="my_qdrant_index", embeddings=embeddings)
        #search_index = Qdrant.from_texts(texts, embeddings, client=client, collection_name="
    return search_index


def similarity_search(query, index):
    # k is the number of similarity searched that matches the query
    # default is 4
    matched_docs = index.similarity_search(query, k=4) 
    sources = []
    for doc in matched_docs:
        sources.append(
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
        )

    return matched_docs, sources

# get the list of pdf files from the docs directory into a list format
pdf_folder_path = 'docs'
doc_list = [s for s in os.listdir(pdf_folder_path) if s.endswith('.pdf')]
num_of_docs = len(doc_list)

# create a loader for the PDFs from the path
loader = PyPDFLoader(os.path.join(pdf_folder_path, doc_list[0]))

start = timer()

# load the documents with Langchain
docs = loader.load()
# Split in chunks
chunks = split_chunks(docs)
# create the db vector index
db0 = create_index(chunks, db_option)

print("Main Vector database created. Start iteration and merging...")
#If you input more than one document, and the Vector DB hasn't been created yet
for i in range(1,num_of_docs):
    print(doc_list[i])
    print(f"loop position {i}")
    loader = PyPDFLoader(os.path.join(pdf_folder_path, doc_list[i]))
    # start = datetime.datetime.now() #not used now but useful
    docs = loader.load()
    chunks = split_chunks(docs)
    dbi = create_index(chunks, "faiss")
    print("start merging with db0...")
    db0.merge_from(dbi)
    # end = datetime.datetime.now() #not used now but useful
    # elapsed = end - start #not used now but useful
    #total time
    # print(f"completed in {elapsed}")
    print("-----------------------------------")

print(f"the daatabase is done with {num_of_docs} subset of db index")
print("-----------------------------------")
print(f"Merging completed")
print("-----------------------------------")
print("Saving Merged Database Locally")
print("-------------------------------------")
print("DB OPTION:", db_option)
# Save the database locally
if db_option == "faiss" or db_option == "FAISS":
    db0.save_local("my_faiss_index")
    print("merged database saved as my_faiss_index")
    print("You can use newdb = FAISS.load_local('my_faiss_index', embeddings) to access it again")
elif db_option == "Qdrant" or db_option == "qdrant" or db_option == "QDRANT":
    print("merged database saved as my_qdrant_index")
    print("You can access this by doing client = qdrant_client.QdrantClient(path='/my_qdrant_index', prefer_grpc=True)")
    print("And then index = Qdrant(client=client, collection_name='my_qdrant_index', embeddings=embeddings)")

end = timer()
print("-----------------------------------")

print("Indexing total time elapsed in seconds:", end-start)
print("-----------------------------------")

print("Now testing just semantic search")

semantic_start = timer()

if db_option == "FAISS" or db_option == "faiss":
    index = FAISS.load_local("my_faiss_index", embeddings)
elif db_option == "qdrant" or db_option == "Qdrant" or db_option == "QDRANT":
    index = db0
    #client = qdrant_client.QdrantClient(path="/my_qdrant_index", prefer_grpc=True)
    #index = Qdrant(client=client, collection_name="my_qdrant_index", embeddings=embeddings)

print("----------------------------------")

#Function to ask a question and run an LLM chain
def askQuestions(vector_store, chain, question):
    #similar_docs = similarity_search(question, index)
    artie_template = """
I'm going to give you a question to answer. If the question is related to the context, use the context to help formulate your answer. If the context is related to the question, you can print and cite the text. If the context is not related to the question, say "I don't know."  Don't answer the question more than once, and do not repeat your answer. You do not need to use all of the space available.
Context: {context}
Question: Answer the following question, if necessary, using the above context in the 'context' variable. {question}
Answer: """
    ethan_template = """ [INSTRUCTIONS]\n You are a helpful chatbot designed to assist the user in various question answering and text generation requests. If you do not know the answer, just say "I don't know". Do not answer the question more than once or repeat your answer. You do not need to use all of the space available for your answer. Your response should be in plaintext. Always use information from the context over your general knowledge if it is relevant to the request. Use this as context for your response: [CONTEXT]\n {context} \n[REQUEST]\n {question}"""
    matched_docs,sources = similarity_search(question, index)
    context = "\n".join([doc.page_content for doc in matched_docs])
    prompt = PromptTemplate(template=artie_template, input_variables=["question", "context"]).partial(context=context)
    llm_chain = LLMChain(prompt=prompt, llm = llm)
    response = llm_chain.run(question)
    # Ask a question using the QA chain
    #similar_docs = vector_store.similarity_search(question)
    #response = chain.run(input_documents=similar_docs, question=question)
    return response

print("Now attempting to chain to an LLM")
print("LLM:", llm)
start=timer()

#Run the chain and ask a question

#Question options for ocean pdf:
q1 = "How many people live in coastal areas?"
q4 = "What are the top 5 biggest concerns to the ocean right now ranked based off of immediate danger. List them from 1 to 5"
q5 = "Who is LeBron James?"
chain = load_qa_chain(llm, chain_type="stuff")

#orthogonality = "What are some techniques to maintain orthogonality?"
answer = askQuestions(index, chain, q4)

end=timer()

print("LLM's ANSWER:")
print(answer + "\n")

total_end = timer()
print("Total time taken for LLM Chain to process:", end-start)
print("------------------------------------------------------")
print("TOTAL TIME ELAPSED IN SECONDS FOR ENTIRE PROCESS:", total_end - total_start)


# ------ This method down here was the one from GPT 4 all using context ------- 
# create the prompt template
#template = """
#Please use the following context to answer questions.
#Context: {context}
#---
#Question: {question}
#Answer: Let's think step by step."""
#start = timer()
# Hardcoded question
#question = "What are some techniques to maintain orthogonality?"
#matched_docs, sources = similarity_search(question, index)
# Creating the context
#context = "\n".join([doc.page_content for doc in matched_docs])
# instantiating the prompt template and the GPT4All chain
#prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)
#llm_chain = LLMChain(prompt=prompt, llm=llm)


