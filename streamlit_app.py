from torch import cuda, bfloat16
import transformers
import streamlit as st

st.title('Tommy bot')

model_id = 'meta-llama/Llama-2-7b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, you need an access token
hf_auth = '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'
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth
)

# enable evaluation mode to allow model inference
model.eval()

print(f"Model loaded on {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

stop_list = ['Human\n\n', '\n```\n']

stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]

import torch

stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
stop_token_ids

from transformers import StoppingCriteria, StoppingCriteriaList

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model rambles during chat
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=256,  # max number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

from langchain.llms import HuggingFacePipeline

llm = HuggingFacePipeline(pipeline=generate_text)

import requests
import xml.etree.ElementTree as ET

# URL of the sitemap.xml file
sitemap_url = "https://www.usc.edu/page-sitemap.xml"

# Send an HTTP request to the sitemap URL
response = requests.get(sitemap_url)

if response.status_code == 200:
    # Parse the XML content of the sitemap
    sitemap_xml = response.text
    root = ET.fromstring(sitemap_xml)

    # Find and extract website URLs
    urls = [element.text for element in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
else:
    print("Failed to retrieve the sitemap.xml")

with open('./urls.txt', "w") as file:
    # Write each item from the list to the file
    for item in urls:
        file.write(item + "\n")

from langchain.document_loaders import WebBaseLoader

web_links = urls

loader = WebBaseLoader(web_links)
documents = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=30)
all_splits = text_splitter.split_documents(documents)

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# storing embeddings in the vector store
vectorstore = FAISS.from_documents(all_splits, embeddings)

retriever = vectorstore.as_retriever()

from operator import itemgetter

from langchain.memory import ConversationBufferMemory

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from langchain.schema import format_document
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableParallel

memory = ConversationBufferMemory(
    return_messages=True, output_key="answer", input_key="question")

# First we add a step to load memory
# This adds a "memory" key to the input object
loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
)

from langchain.prompts.prompt import PromptTemplate

_template = """Combine the chat history and follow up question into a standalone question in its original language.

Chat History:
{chat_history}
Follow up question: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

# Now we calculate the standalone question
standalone_question = {
    "standalone_question": {
        "question": lambda x: x["question"],
        "chat_history": lambda x: get_buffer_string(x["chat_history"]),
    }
    | CONDENSE_QUESTION_PROMPT
    | llm
    | StrOutputParser(),
}

# Now we retrieve the documents
retrieved_documents = {
    "docs": itemgetter("standalone_question") | retriever,
    "question": lambda x: x["standalone_question"],
}

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

# Now we construct the inputs for the final prompt
final_inputs = {
    "context": lambda x: _combine_documents(x["docs"]),
    "question": itemgetter("question"),
}

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

# And finally, we do the part that returns the answers
answer = {
    "answer": final_inputs | ANSWER_PROMPT | llm,
    "docs": itemgetter("docs"),
}

# And now we put it all together!
final_chain = loaded_memory | standalone_question | retrieved_documents | answer

inputs = {"question": "What the dorm options avaiable for freshman?"}
result = final_chain.invoke(inputs)

print(result['answer'])





