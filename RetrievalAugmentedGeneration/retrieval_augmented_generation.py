
#%%
import os

import hvac

from datetime import date, timedelta

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from langchain.chains import RetrievalQA, create_qa_with_sources_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StdOutCallbackHandler

from langchain.docstore.document import Document

from newsapi import NewsApiClient
#%% 
# Retrieve the openai_test_token and the newsapi_token from Vault
client = hvac.Client(url='http://localhost:8200')
client.token = os.environ['VAULT_TOKEN']

#%%
if not client.is_authenticated():
    print('Client is not authenticated.')
    exit(1)

try:
    openai_token = client.secrets.kv.v2.read_secret(path='openai', mount_point='kv')['data']['data']['openai_test_token']
except hvac.exceptions.InvalidPath:
    print('The openai secret path is invalid.')
    exit(1)

#%%
try:
    newsapi_token = client.secrets.kv.v2.read_secret(path='news_api', mount_point='kv')['data']['data']['news_api_key']
except hvac.exceptions.InvalidPath:
    print('The openai secret path is invalid.')
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

handler = StdOutCallbackHandler()

chain.run(
    'What is machine learning?',
    callbacks=[handler]
)
# %%
newsapi = NewsApiClient(api_key=newsapi_token)

today = date.today()
last_week = today - timedelta(days=7)

## Get the news on 'Artificial Intelligence' from last week
latest_news = newsapi.get_everything(
    q='artificial intelligence',
    from_param=last_week.strftime('%Y-%m-%d'),
    to=today.strftime('%Y-%m-%d'),
    sort_by='relevancy',
    language='en'
)

# %%
## Create documents 
docs = [
    Document(
        page_content=article['title'] + '\n\n' + article['description'] if article['description'] else '',
        metadata = {
            'source': article['url']
        }
    ) for article in latest_news['articles']
]
# %%
## Create a chain that provides the sources with the answers
qa_chain = create_qa_with_sources_chain(llm)

doc_prompt = PromptTemplate(
    template='Content: {page_content}\nSource: {source}',
    input_variables=['page_content', 'source'],
)

final_qa_chain = StuffDocumentsChain(
    llm_chain=qa_chain,
    document_variable_name='context',
    document_prompt=doc_prompt,
)

index = FAISS.from_documents(docs, embedding=embeddings)

chain = RetrievalQA(
    retriever=index.as_retriever(),
    combine_documents_chain=final_qa_chain
)

# %%
## Ask a question
question = """
What is the most important news about artificial intelligence from last week?
"""
# %%
answer = chain.run(question)
print(answer)
# %%
