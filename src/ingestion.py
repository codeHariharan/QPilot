import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from pathlib import Path

class Ingest:

    def process_doc(self, doc_directory):
        doc_dir = Path(doc_directory)
        pdf_files = list(doc_dir.glob("**/*.pdf"))
        txt_files = list(doc_dir.glob("**/*.txt"))
        all_docs=[]
        if pdf_files:
            try:
                for pdf in pdf_files:
                    loader = PyPDFLoader(str(pdf))
                    documents = loader.load()
        
                for doc in documents:
                    doc.metadata['source_file'] = pdf.name
                    doc.metadata['file_type'] = 'pdf'
                
                all_docs.extend(documents)
            except Exception as e:
                print(f"✗ Error: {e}")
        if txt_files:
            try:
                for txt in txt_files:
                    loader = TextLoader(str(txt))
                    documents = loader.load()
                        
                for doc in documents:
                    doc.metadata['source_file'] = txt.name
                    doc.metadata['file_type'] = 'txt'
                                
                all_docs.extend(documents)
            except Exception as e:
                print(f"✗ Error: {e}")
       # print(f"\nTotal documents loaded: {len(all_docs)}")
        return all_docs
