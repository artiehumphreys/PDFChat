from langchain import HuggingFaceHub, LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import GPT4All
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import PyPDFLoader
import datetime
import os
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores.faiss import FAISS
import pandas as pd
import faiss

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

llms = ['gpt2-large', 'tiiuae/falcon-7b-instruct', 'openlm-research/open_llama_3b']

#embedding model options:
MPNET_MODEL = "sentence-transformers/all-mpnet-base-v2"
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MULTIQA_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

#Specify the embedding model here
model_name = MULTIQA_MODEL

#Create the HuggingFaceEmbeddings object
embeddings = HuggingFaceEmbeddings(model_name=model_name)

inp = input("Pick which LLM to use. Type 0 for GPT-2, type 1 for Falcon-7B (Recommended for Document QA and fast responses), and type 2 for OpenLLama" + "\n")

if (not inp.strip().isdigit()):
    inp = input("Please enter a valid number here: ")

int_inp = int(inp.strip())

if (int_inp > 3 or int_inp < 0):
    inp = input("Please enter a valid number here: ")

repo_id = llms[int_inp]

callback_manager = AsyncCallbackManager([StreamingStdOutCallbackHandler()])

if (int_inp == 4):
    gpt4all_path = './test/models/ggml-gpt4all-j-v1.3-groovy.bin'
    llm = GPT4All(model=gpt4all_path, callback_manager=callback_manager, verbose=True)
else:   
    llm = HuggingFaceHub(
        repo_id = repo_id,
        model_kwargs={'temperature':0.5, 'max_new_tokens':500, 'repition_penalty':0.5}
    )   
    # check_llm = HuggingFaceHub(
    #     repo_id = repo_id,
    #     model_kwargs={'temperature':0.5, 'max_new_tokens':500, 'repition_penalty':0.5}
    # )   
#Context: {context}
#{question}
#Sources: {sources}
# Please use the following context to answer questions. Please add the context to the response if needed.
template = """
I'm going to give you a question to answer. Answer it to the best of your ability and say that you don't know if you don't have an answer. If the question is related to the context, use the context to help formulate your answer. If the context is related to the question, you can print and cite the text using the 'sources' variable below.
Here are some general rules to follow:
1. If you are asked to rank something or make a list, you will only make a list or rank with the number of items specified. If you are asked to rank something 1 through 5, you will rank only five items.
2. You may use your general knowledge to answer questions, but first try and answer the question using the provided context.
3. If you are asked to summarize anything from the context, do the best you can. Try to encapsulate the main idea behind the context.
4. Use logic to answer questions if necessary.

Context: {context}
Question: Answer the following question in paragraph format, if necessary, using the above context in the 'context' variable. {question}
Answer: """

check_template = """
You are going to use your logic skills to evaluate this claim and see if it is accurate. If it is not accurate. Provide a better answer. 
Response: {answer}
Answer: """

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
text = []
def split_chunks(sources):
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=32)
    for chunk in splitter.split_documents(sources):
        chunks.append(chunk)
    return chunks

def create_index(chunks):
    global text
    data = []

    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    # data.append(texts)
    # df = pd.DataFrame(texts, columns = ['text'])
    # text = df['text'] 
    # vectors = model.encode(text)
    # vector_dimension = vectors.shape[1]
    # index = faiss.IndexFlatL2(vector_dimension)
    # faiss.normalize_L2(vectors)
    # index.add(vectors) 
    search_index = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    #embeddings = model.encode(texts)
    #text_embeddings = zip(texts, embeddings)
    #search_index = FAISS.from_embeddings(text_embeddings=text_embeddings, embedding = model)
    return search_index


def similarity_search(query, index):
    # k is the number of similarity searched that matches the query
    # default is 4
    matched_docs = index.similarity_search(query, k=3) 
    sources = []
    for doc in matched_docs:
        sources.append(
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
        )

    return matched_docs, sources

pdf_folder_path = 'docs'
doc_list = [s for s in os.listdir(pdf_folder_path) if s.endswith('.pdf')]
num_of_docs = len(doc_list)
# create a loader for the PDFs from the path
loader = PyPDFLoader(os.path.join(pdf_folder_path, doc_list[0]))
# load the documents with Langchain
docs = loader.load()
# Split in chunks
chunks = split_chunks(docs)
# create the db vector index
db0 = create_index(chunks)

print("Main Vector database created. Start iteration and merging...")
for i in range(0,num_of_docs):
    print(doc_list[i])
    print(f"loop position {i}")
    loader = PyPDFLoader(os.path.join(pdf_folder_path, doc_list[i]))
    start = datetime.datetime.now() #not used now but useful
    docs = loader.load()
    chunks = split_chunks(docs)
    dbi = create_index(chunks)
    print("start merging with db0...")
    db0.merge_from(dbi)
    end = datetime.datetime.now() #not used now but useful
    elapsed = end - start #not used now but useful
    #total time
    print(f"completed in {elapsed}")
    print("-----------------------------------")
# loop_end = datetime.datetime.now() #not used now but useful
# loop_elapsed = loop_end - loop_start #not used now but useful
# print(f"All documents processed in {loop_elapsed}")
print(f"the database is done with {num_of_docs} subset of db index")
print("-----------------------------------")
print(f"Merging completed")
print("-----------------------------------")
print("Saving Merged Database Locally")
# Save the databasae locally
db0.save_local("my_faiss_index")
#faiss.write_index(db0, "my_faiss_index")

question = input("Ask the model a question here: ")

index = FAISS.load_local("my_faiss_index", embeddings)
#docs2 = index.similarity_search(question)
matched_docs, nembo = similarity_search(question, index)
context = "\n".join([doc.page_content for doc in matched_docs])

# def askQuestions(vector_store, chain, question):
#     # Ask a question using the QA chain
#     similar_docs = vector_store.similarity_search(question)
#     response = chain.run(input_documents=similar_docs, question=question)
#     return response

similar_docs = index.similarity_search(question)

prompt = PromptTemplate(template=template, input_variables=["question"], partial_variables = {'context' : context})

start = datetime.datetime.now()

llm_chain = LLMChain(prompt=prompt, llm = llm)

answer = llm_chain.run(question)
if (int_inp == 1): print (answer[0] + "\n")
else: print("LLM Answer: " + answer + "\n")

elapsed = datetime.datetime.now() - start
print("The " + llms[int_inp] + " model processed and answered the query in " + str(elapsed))

# check_prompt = PromptTemplate(template=check_template, input_variables=["answer"]).partial(answer = answer)

# llm_check_chain = LLMChain(prompt = check_prompt, llm = check_llm)

# answer2 = llm_check_chain.run()

# print("LLM fact-check: " + answer2)
