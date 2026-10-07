from sentence_transformers import SentenceTransformer


class EmbeddingManager:

    def __init__(self, model_name: str ):

        self.model_name = model_name
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimension: {self.model.get_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: list[str]):
        try:
            if not self.model:
             raise ValueError("Model not loaded")
            embeddings = self.model.encode(
                    texts,
                    batch_size=64,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                )
            #print(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
                print(f"✗ Error: {e}")
        return embeddings