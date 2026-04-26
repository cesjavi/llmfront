import uuid
import logging
import math
from collections import Counter
from io import BytesIO
import pypdf

logger = logging.getLogger(__name__)

# --- Pure Python TF-IDF / BM25 Implementation ---
class SimpleBM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.documents = []
        self.metadatas = []

    def _tokenize(self, text):
        return text.lower().replace('\n', ' ').split()

    def add_documents(self, docs, metadatas):
        for i, doc in enumerate(docs):
            tokens = self._tokenize(doc)
            self.documents.append(doc)
            self.metadatas.append(metadatas[i])
            self.doc_len.append(len(tokens))
            freq = Counter(tokens)
            self.doc_freqs.append(freq)
            for word in freq:
                self.idf[word] = self.idf.get(word, 0) + 1
            self.corpus_size += 1

        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 0
        
        # Precompute IDF
        for word, freq in self.idf.items():
            self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def search(self, query, top_k=3):
        if self.corpus_size == 0:
            return []
        tokens = self._tokenize(query)
        scores = [0.0] * self.corpus_size
        for i in range(self.corpus_size):
            score = 0.0
            doc_len = self.doc_len[i]
            freqs = self.doc_freqs[i]
            for word in tokens:
                if word not in freqs:
                    continue
                f = freqs[word]
                numerator = self.idf.get(word, 0) * f * (self.k1 + 1)
                denominator = f + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += numerator / denominator
            scores[i] = score
            
        top_indices = sorted(range(self.corpus_size), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for i in top_indices:
            if scores[i] > 0:
                results.append((self.documents[i], self.metadatas[i], scores[i]))
        return results

    def count(self):
        return self.corpus_size

RAG_ENABLED = True
collection = SimpleBM25()

def extract_text_from_file(filename: str, content: bytes) -> str:
    ext = filename.split('.')[-1].lower()
    if ext == 'pdf':
        try:
            reader = pypdf.PdfReader(BytesIO(content))
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text
        except Exception as e:
            logger.error(f"Error reading PDF {filename}: {e}")
            return ""
    else:
        try:
            return content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                return content.decode('latin-1')
            except Exception as e:
                logger.error(f"Error decoding text file {filename}: {e}")
                return ""

def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 300) -> list[str]:
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

def process_and_store_document(filename: str, content: bytes) -> dict:
    if not RAG_ENABLED:
        return {"error": "RAG no está disponible."}
        
    text = extract_text_from_file(filename, content)
    if not text.strip():
        return {"error": "No se pudo extraer texto del archivo o está vacío."}
        
    doc_id = str(uuid.uuid4())
    chunks = chunk_text(text, chunk_size=1500, overlap=300)
    
    metadatas = [{"source": filename, "doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]
    
    try:
        collection.add_documents(chunks, metadatas)
        return {
            "doc_id": doc_id,
            "filename": filename,
            "chunks_count": len(chunks)
        }
    except Exception as e:
        logger.error(f"Add document error: {e}")
        return {"error": str(e)}

def query_rag_context(query: str, n_results: int = 3) -> str:
    if not RAG_ENABLED or collection.count() == 0:
        return ""
        
    try:
        results = collection.search(query, top_k=n_results)
        if results:
            context_pieces = []
            for doc, meta, score in results:
                source = meta.get('source', 'Documento')
                context_pieces.append(f"--- Fragmento de {source} ---\n{doc}")
            return "\n\n".join(context_pieces)
        return ""
    except Exception as e:
        logger.error(f"Query error: {e}")
        return ""
