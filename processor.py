from collections import OrderedDict
from typing import Union, List, Optional, Literal

from haystack import Document
from haystack.nodes import PreProcessor


class SkimlinksPreProcessor(PreProcessor):
    def process(
            self,
            documents: Union[dict, Document, List[Union[dict, Document]]],
            clean_whitespace: Optional[bool] = True,
            clean_header_footer: Optional[bool] = True,
            clean_empty_lines: Optional[bool] = None,
            remove_substrings: Optional[List[str]] = None,
            split_by: Optional[Literal["word", "sentence", "passage"]] = None,
            split_length: Optional[int] = None,
            split_overlap: Optional[int] = None,
            split_respect_sentence_boundary: Optional[bool] = None,
            id_hash_keys: Optional[List[str]] = None,
    ) -> List[Document]:

        mapped_values = OrderedDict()
        mapped_values['merchant_name'] = "This business "
        mapped_values['title'] = "sells the product "
        mapped_values['brand'] = "of brand "
        mapped_values['regprice'] = "the price is "
        mapped_values['cat_mapping'] = "the item has been categorized as "
        mapped_values['url'] = "the product can be found in the url "
        mapped_values['descr'] = "the url describes the product as "

        c = ""
        for key, values in mapped_values.items():
            if key in documents.meta:
                if 'price' in key:
                    c += f"{mapped_values[key]} {int(documents.meta[key]) / 100}. "
                else:
                    c += f"{mapped_values[key]} {documents.meta[key]}. "
        documents.content += ' ' + c
        return super().process(documents, id_hash_keys=id_hash_keys)
