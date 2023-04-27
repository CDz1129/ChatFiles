import uuid
from file import (check_index_file_exists,
                  get_index_name_from_file_path, get_index_name_from_compress_filepath)
from llm import get_index_by_index_name, create_index, create_github_index, create_graph, get_graph_by_graph_name
from prompt import get_prompt
from urllib.parse import urlparse


def check_llama_index_exists(file_name):
    index_name = get_index_name_from_file_path(file_name)
    return check_index_file_exists(index_name)


def create_llama_index(filepath):
    index_name = get_index_name_from_file_path(filepath)
    index = create_index(filepath, index_name)
    return index_name, index


def create_giturl_index(giturl):
    parsed_url = urlparse(giturl)
    # 获取 owner 和 repo 名称
    owner = parsed_url.path.split('/')[1]
    repo = parsed_url.path.split('/')[2]
    index_name = owner + repo
    index = create_github_index(owner, repo, index_name)
    return index_name, index


def create_llama_graph_index(filepaths):
    index_set = {}
    for filepath in filepaths:
        index_name = get_index_name_from_compress_filepath(filepath)
        index = create_index(filepath, index_name)
        index_set[index_name] = index
    graph_name = str(uuid.uuid4())
    graph = create_graph(index_set, graph_name)
    return graph_name, graph


def get_answer_from_index(text, index_name):
    index = get_index_by_index_name(index_name)
    return index.query(text, text_qa_template=get_prompt())


def get_answer_from_index_stream(message, index_name):
    index = get_index_by_index_name(index_name)
    return index.query(message, streaming=True, text_qa_template=get_prompt())


def get_answer_from_graph(text, graph_name):
    graph = get_graph_by_graph_name(graph_name)
    return graph.query(text)
