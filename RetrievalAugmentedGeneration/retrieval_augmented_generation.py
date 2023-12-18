
#%%
import os

import hvac
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StdOutCallbackHandler

# Retrieve the openai_test_token from Vault
client = hvac.Client(url='http://localhost:8200')
client.token = os.environ['VAULT_TOKEN']

#%%
if not client.is_authenticated():
    print('Client is not authenticated.')
    exit(1)

try:
    openai_token = client.secrets.kv.v2.read_secret(path='openai', mount_point='kv')['data']['data']['openai_test_token']
     #.v2.read_secret(path='secrets/api_readonly/openai_test_token')
except hvac.exceptions.InvalidPath:
    print('The secret path is invalid.')
    exit(1)


file_path = './ESLII_print12_toc.pdf'

#%%
## Load the book pdf
loader = PyPDFLoader(file_path=file_path)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0
)

data = loader.load_and_split(text_splitter=text_splitter)
data


# %%
embeddings = OpenAIEmbeddings(openai_api_key=openai_token, show_progress_bar=True)
vector1 = embeddings.embed_query('How are you?')
len(vector1)
# %%
## Embed the book data
index = FAISS.from_documents(data, embeddings)

index.similarity_search_with_relevance_scores(
    'What is machine learning?'
)

# %%
retriever = index.as_retriever()
retriever.search_kwargs['fetch_k'] = 20
retriever.search_kwargs['maximal_marginal_relevance'] = True
retriever.search_kwargs['k'] = 10

llm = ChatOpenAI(openai_api_key=openai_token)

chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    verbose=True
)

handler = StdOutCallbackHandlerq()

chain.run(
    'What is machine learning?',
    callbacks=[handler]
)