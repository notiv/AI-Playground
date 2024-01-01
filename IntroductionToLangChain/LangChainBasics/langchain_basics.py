
import hvac
import os

from langchain.chains import ConversationChain, LLMChain, SimpleSequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import BaseOutputParser

# Retrieve the openai_test_token from Vault
client = hvac.Client(url='http://localhost:8200')
client.token = os.environ['VAULT_TOKEN']

if not client.is_authenticated():
    print('Client is not authenticated.')
    exit(1)

try:
    openai_token = client.secrets.kv.v2.read_secret(path='openai', mount_point='kv')['data']['data']['openai_test_token']
     #.v2.read_secret(path='secrets/api_readonly/openai_test_token')
except hvac.exceptions.InvalidPath:
    print('The secret path is invalid.')
    exit(1)


chat_model = ChatOpenAI(openai_api_key=openai_token)

reply1 = chat_model.predict('How are you?')
reply2 = chat_model.predict('What was my previous question?')
print(reply1)
print(reply2)

## Conversation Chain

chain = ConversationChain(
    llm=chat_model,
    verbose=True
)

reply3 = chain.run('How are you today?')
reply4 = chain.run('What was my previous question?')
print(reply3)
print(reply4)

## Prompt Templates
template = """
    Return all the subcategories of the following category: 
    
    {category}
"""

prompt = PromptTemplate(
    input_variables=['category'],
    template=template,
)

chain = LLMChain(
    llm=chat_model,
    prompt=prompt,
    verbose=True
)

reply5 = chain.run('Machine Learning')
print(reply5)

## System & Human Prompts
system_template = """
You are a helpful assistant who generates comma separated lists.
A user will only pass a category and you should generate subcategories of that category in a comma separated list.
ONLY return comma separated lists and nothing more.
"""

human_template = '{category}'

system_message = SystemMessagePromptTemplate.from_template(system_template)
human_message = HumanMessagePromptTemplate.from_template(human_template)

prompt = ChatPromptTemplate.from_messages([system_message, human_message])

chain = LLMChain(
    llm=chat_model,
    prompt=prompt,
    verbose=True
)

reply6 = chain.run('Machine Learning')
print(reply6)

## Output Parser
class CommaSeparatedParser(BaseOutputParser):
    def parse(self, text):
        output = text.strip().split(',')
        output = [o.strip() for o in output]
        return output
    
chain = LLMChain(
    llm=chat_model,
    prompt=prompt,
    output_parser=CommaSeparatedParser(),
    verbose=True
)

reply7 = chain.run('Machine Learning')
print(reply7)

## Several Inputs
input_list = [
    {'category': 'food'},
    {'category': 'country'},
    {'category': 'colors'},
]

response = chain.apply(input_list)
print(response[2]['text'])

## Simple Sequence

title_template = """
You are a writer. Given a subject, your job is to return a fun title for a play.

Subject: {subject}
Title:"""

title_chain = LLMChain.from_string(
    llm=chat_model,
    template=title_template,
)

title_chain.run('Machine Learning')

synopsis_template = """
You are a writer.
Given a title, write a synopsis for a play.

Title: {title}
Synopsis:"""

synopsis_chain = LLMChain.from_string(
    llm=chat_model,
    template=synopsis_template
)

title = 'The Algorithmic Adventure: A Machine Learning Marvel'

synopsis_chain.run(title)

### Combining Chains
chain = SimpleSequentialChain(
    chains=[title_chain, synopsis_chain],
    verbose=True
)

reply8 = chain.run('Machine Learning')
print(reply8)

pass