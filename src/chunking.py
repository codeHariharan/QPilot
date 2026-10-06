from langchain_text_splitters import RecursiveCharacterTextSplitter
import yaml
from pathlib import Path

class Chunking:

    def split(self, documents:list[any]):
        try:
            config_path = Path(__file__).resolve().parents[1] / "config" / "config.yml"

            with config_path.open(encoding="utf-8") as config_file:
                config = yaml.safe_load(config_file)

            chunking = config["chunking"]
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = chunking['chunk_size'],
                chunk_overlap = chunking['chunk_overlap'],
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            split_docs = text_splitter.split_documents(documents)
        except Exception as e:
            print(f"✗ Error: {e}")
        return split_docs


