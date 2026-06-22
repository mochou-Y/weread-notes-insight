import numpy as np
import tempfile
import unittest

from src.embedding.embedder import Embedder, EmbeddingStorage


class DummyEmbedder(Embedder):
    def __init__(self):
        self.calls = []

    def embed_texts(self, texts, batch_size=None):
        self.calls.extend(texts)
        return [[float(len(text))] for text in texts]


class EmbeddingResumeTest(unittest.TestCase):
    def test_embed_notes_resumes_from_saved_partial_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = EmbeddingStorage(tmp_dir)
            storage.save_partial(np.array([[1.0], [2.0]]))

            embedder = DummyEmbedder()
            notes = [type("Note", (), {"content": text, "context": None}) for text in ["a", "bb", "ccc"]]

            embeddings = embedder.embed_notes(notes, storage=storage)

            self.assertEqual(embedder.calls, ["ccc"])
            self.assertEqual(embeddings.tolist(), [[1.0], [2.0], [3.0]])


if __name__ == "__main__":
    unittest.main()
