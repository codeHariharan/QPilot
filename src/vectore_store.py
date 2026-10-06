import os
import chromadb
import hashlib
import numpy as np

class VectorStore:
    def __init__(self, collection_name:str, persistant_dir: str):
        self.collection_name = collection_name
        self.collection = None
        self.client = None
        self.persistant_dir = persistant_dir
        self.intialize_store()

    def intialize_store(self):
        try:
            os.makedirs(self.persistant_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persistant_dir)
            self.collection = self.client.get_or_create_collection(
                name = self.collection_name,
                metadata={"description": "User document embeddings for RAG"}
            )
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: list[any], embeddings: np.ndarray):
       
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []
        occurrences = {}
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
           
            metadata = dict(doc.metadata)
            source = metadata.get("source", metadata.get("source_file", ""))
            page = metadata.get("page", "")
            content = doc.page_content
            fingerprint = f"{source}\0{page}\0{content}"
            occurrence = occurrences.get(fingerprint, 0)
            occurrences[fingerprint] = occurrence + 1
            id_hash = hashlib.sha256(
                f"{fingerprint}\0{occurrence}".encode("utf-8")
            ).hexdigest()
            ids.append(f"doc_{id_hash}")

            metadata['doc_index'] = i
            metadata['content_length'] = len(content)
            metadatas.append(metadata)
            
            documents_text.append(content)
         
            embeddings_list.append(embedding.tolist())
        
        
        try:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            #print(f"Successfully added or updated {len(documents)} documents in vector store")
            #print(f"Total documents in collection: {self.collection.count()}")
            
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise