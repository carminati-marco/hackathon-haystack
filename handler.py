from haystack import Pipeline
from haystack.document_stores import SQLDocumentStore
from haystack.document_stores import InMemoryDocumentStore
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.nodes import FARMReader, TfidfRetriever, BM25Retriever, EmbeddingRetriever
from haystack.pipelines import ExtractiveQAPipeline

from pprint import pprint
import config
from processor import SkimlinksPreProcessor

#----
# document_store = InMemoryDocumentStore()
# document_store = SQLDocumentStore(url="sqlite:///stg.sqlite")
# document_store = SQLDocumentStore(url="mysql://root:root@127.0.0.1:3306/rm_stage")
document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="rm_index")
#----
# retriever = TfidfRetriever(document_store=document_store)
# retriever2 = BM25Retriever(document_store=document_store)
retriever3 = EmbeddingRetriever(document_store=document_store,
                                embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
                                model_format="sentence_transformers",
                                top_k=20)
# ------
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

# ------
pipe = Pipeline()
# pipe.add_node(component=retriever, name="TfidfRetriever", inputs=["Query"])
# pipe.add_node(component=retriever2, name="BM25Retriever", inputs=["Query"])
pipe.add_node(component=retriever3, name="EmbeddingRetriever", inputs=["Query"])

pipe.add_node(component=reader, name="FARMReader", inputs=["EmbeddingRetriever"])

processor = SkimlinksPreProcessor()

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
        verify_certs=True,
        scheme=config.ELASTIC_SCHEME,
    )

    document_store.update_embeddings(retriever3)

    return {"ok": True}


def query_haystack(query) -> dict:
    prediction = pipe.run(
        query=query,
        params={"EmbeddingRetriever": {"top_k": 10}, "FARMReader": {"top_k": 5}}
    )

    pprint("*" * 10)
    pprint("query: {}".format(prediction['query']))
    pprint("*" * 10)
    i=0
    for item in prediction['answers']:
        pprint("score: {}".format(item.score))
        pprint("answer: {}".format(item.answer))
        pprint("context: {}".format(item.context))
        pprint("search_id: {}".format((item.meta["search_id"])))
        pprint("-"*10)
        i+=1
        if i==3:
            break
