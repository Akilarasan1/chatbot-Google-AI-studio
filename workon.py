# This is a long document we can split up.
from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("../../state_of_the_union.txt") as f:
    state_of_the_union = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 100,
        chunk_overlap = 20, 
        length_function = len,
        is_separator_regex = False
    )

    texts = text_splitter.create_documents([state_of_the_union])
    print(texts[0])
    print(texts[1])

    text_splitter.split_text(state_of_the_union)[:2]