#%%
import hvac
import os
import pandas as pd
import random

#%%
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain, LLMChain
from langchain.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate

#%%
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
# %%
chat_model = ChatOpenAI(openai_api_key=openai_token)
# %%
conversation_chain = ConversationChain(
    llm=chat_model
)

conversation_chain.prompt.template
# %%
# ## Few Shot Learning
examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2+3", "output": "5"},
    {"input": "5+6", "output": "11"},
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

print(few_shot_prompt.format())
# %%
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a wizard of math."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)
# %%
print(final_prompt.format(input="What is 3 + 3"))
# %%
chain = LLMChain(
    llm=chat_model,
    prompt=final_prompt,
    verbose=True
)

chain.run("What is 3 + 3")
# %%
# ## Memetic Proxy
template = """
System: {reference}
Provide a helpful response to the following question

Human: {question}

AI:"""

prompt = PromptTemplate.from_template(template)

chain = LLMChain(
    llm=chat_model,
    prompt=prompt,
    verbose=True
)

high_level = 'Imagine you are a Professor teaching at a PhD level'
lower_level = 'Imagine you are a Kindergarten teacher.'

# %%
question = 'What is Quantum Mechanics?'

chain.run({
    'question': question,
    'reference': high_level
})
# %%
chain.run({
    'question': question,
    'reference': lower_level
})

# %%
# ## Self-consistency
chat_model = ChatOpenAI(openai_api_key=openai_token, temperature=0.7)
# %%
cot_chain = LLMChain(
    llm=chat_model,
    prompt=final_prompt,
    verbose=True
)