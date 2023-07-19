from haystack.document_stores import SQLDocumentStore
from haystack.nodes import FARMReader, TfidfRetriever
from haystack.pipelines import ExtractiveQAPipeline

import config
from processor import SkimlinksPreProcessor

document_store = SQLDocumentStore(url="sqlite:///stg.sqlite")
retriever = TfidfRetriever(document_store=document_store)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
pipe = ExtractiveQAPipeline(reader, retriever)

processor = SkimlinksPreProcessor(
    # clean_empty_lines=True,
    # clean_whitespace=True,
    # clean_header_footer=True,
    # remove_substrings=True,
    # split_by="word",
    # split_length=200,
    # split_respect_sentence_boundary=True,
    # split_overlap=0,
    # max_chars_check=10000
)


def setup_haystack():
    from haystack.document_stores import elasticsearch_index_to_document_store
    elasticsearch_index_to_document_store(
        document_store=document_store,
        original_content_field="title",
        original_index_name=config.ELASTIC_INDEX,
        original_name_field="title",
        preprocessor=processor,
        port=config.ELASTIC_PORT,
        host=config.ELASTIC_CLOUD_HOST,
        api_key_id=config.ELASTIC_API_ID,
        api_key=config.ELASTIC_API_KEY,
        verify_certs=False,
        scheme=config.ELASTIC_SCHEME,
    )
    return {"ok": True}


def query_haystack(query) -> dict:
    prediction = pipe.run(
        query=query,
        params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
    )
    from pprint import pprint
    pprint(prediction)
