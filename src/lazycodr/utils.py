import json
from functools import wraps
from pathlib import Path

import httpx
import requests
from github import Github
from langchain import LLMChain, PromptTemplate
from langchain.chains.combine_documents.refine import RefineDocumentsChain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import TokenTextSplitter
from rich.console import Console

from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from lazycodr.constants import (
    PR_REFINE_INIT_TEMPLATE_NAME,
    PR_REFINE_LOOP_TEMPLATE_NAME,
)
from lazycodr.prompts import load_template

console = Console()


def check_credentials():
    """Check if credentials file exists"""
    global credentials
    try:
        with open(Path.home() / ".lazy-coder-credentials.json") as json_file:
            cred = json.load(json_file)
            credentials = cred
            return cred
    except Exception:
        raise Exception("Credentials file not found")


# Decorator that loads and provides credentials to the function
# use funcools.wraps to preserve the function name and docstring
def use_credentials(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        credentials = check_credentials()
        return func(credentials, *args, **kwargs)

    return wrapper


@use_credentials
def get_pr_diff(credentials, repo_name, pr_number):
    g = Github(credentials["github_token"])
    repo = g.get_repo(repo_name)
    pr = repo.get_pull(pr_number)
    diff_content = ""

    for file in pr.get_files():
        if file.patch:
            diff_content += file.patch

    return diff_content, pr


@use_credentials
def generate_pr(credentials, pr_diff: str, pr_template: str):
    llm = Ollama(model="llama2",
                 callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))


    text_splitter = TokenTextSplitter(chunk_size=4000, chunk_overlap=200)
    docs = text_splitter.create_documents([pr_diff])

    refine_prompt_template = load_template(PR_REFINE_LOOP_TEMPLATE_NAME).render(
        pr_template=pr_template
    )
    refine_prompt = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_prompt_template,
    )

    prompt_template = load_template(PR_REFINE_INIT_TEMPLATE_NAME).render(
        pr_template=pr_template
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    initial_chain = LLMChain(llm=llm, prompt=prompt)
    _refine_llm = llm
    refine_chain = LLMChain(llm=_refine_llm, prompt=refine_prompt)

    chain = RefineDocumentsChain(
        initial_llm_chain=initial_chain,
        refine_llm_chain=refine_chain,
        document_variable_name="text",
        initial_response_name="existing_answer",
    )

    print(f"{len(docs)} splits to process ...")

    res = chain.run(docs)
    return res
