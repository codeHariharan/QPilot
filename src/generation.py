from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

class RAGRetriever:
    def __init__(self, vector_store , embedding_manager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query, top_k: int=5):
        query_embedding= self.embedding_manager.generate_embeddings([query])[0]
       
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            
            retrieved_docs = []
            
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
                
                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    
                    similarity_score = 1 - distance
                    
                    retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })
                
                #print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found")
            
            return retrieved_docs
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
    def generate(self,query,retriever,top_k=3):
        try:
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY is not set")
            llm = ChatGroq(
                api_key = groq_api_key,
                model = "openai/gpt-oss-20b",
                temperature = 0.1,
                max_tokens = 1024
            )
            results=retriever.retrieve(query,top_k=top_k)
            context="\n\n".join([doc['content'] for doc in results]) if results else ""
            if not context:
                return "No relevant context found to answer the question."
            prompt =f"""Use the following context to answer the question concisely.
                Context:
                {context}

                Question: {query}

                Answer:"""
    
            response=llm.invoke([prompt.format(context=context,query=query)])
            response = llm.invoke(prompt.format(context=context,query=query))
        except Exception as e:
                print(f"Error during retrieval: {e}")
        return response.content