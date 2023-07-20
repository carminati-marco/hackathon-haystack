from typing import Union, List, Optional, Literal

from haystack import Document
from haystack.nodes import PreProcessor


class SkimlinksPreProcessor(PreProcessor):
    def process(
            self,
            documents: Union[dict, Document, List[Union[dict, Document]]],
            clean_whitespace: Optional[bool] = None,
            clean_header_footer: Optional[bool] = None,
            clean_empty_lines: Optional[bool] = None,
            remove_substrings: Optional[List[str]] = None,
            split_by: Optional[Literal["word", "sentence", "passage"]] = None,
            split_length: Optional[int] = None,
            split_overlap: Optional[int] = None,
            split_respect_sentence_boundary: Optional[bool] = None,
            id_hash_keys: Optional[List[str]] = None,
    ) -> List[Document]:
        # print(documents)
        c = ""
        for i, j in documents.meta.items():
            if 'price' in i:
                c += f"{i} {int(j) / 100}\n"
            else:
                c += f"{i} {j}\n"
        documents.content += ' ' + c
        return super().process(documents, id_hash_keys=id_hash_keys)
