"""
BM25 Prototype
==============

Basic BM25 indexing prototype for text-based sparse retrieval.
This demonstrates BM25 capability even though Kalibry Finance
uses regime-based filtering instead of text-based BM25.

Usage:
    python -m src.prototypes.bm25_prototype
"""

from rank_bm25 import BM25Okapi
from typing import List, Tuple, Dict
import numpy as np


class BM25Index:
    """
    Simple BM25 index for text document retrieval
    
    BM25 (Best Matching 25) is a ranking function used by search engines
    to estimate the relevance of documents to a given search query.
    
    Formula:
        score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D|/avgdl))
    
    where:
        - f(qi, D) = frequency of term qi in document D
        - |D| = length of document D
        - avgdl = average document length
        - k1, b = free parameters (typically k1=1.5, b=0.75)
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 index
        
        Args:
            k1: Term frequency saturation parameter (default 1.5)
            b: Document length normalization parameter (default 0.75)
        """
        self.k1 = k1
        self.b = b
        self.bm25: BM25Okapi = None
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.tokenized_docs: List[List[str]] = []
        self.is_built = False
    
    def tokenize(self, text: str) -> List[str]:
        """
        Simple whitespace tokenization with lowercasing
        
        For production, use proper tokenizers like:
        - nltk.word_tokenize
        - spacy tokenizer
        - transformers tokenizer
        """
        return text.lower().split()
    
    def build(self, documents: List[str], doc_ids: List[str] = None):
        """
        Build BM25 index from documents
        
        Args:
            documents: List of document texts
            doc_ids: Optional list of document IDs (defaults to indices)
        """
        self.documents = documents
        self.doc_ids = doc_ids or [str(i) for i in range(len(documents))]
        
        # Tokenize documents
        self.tokenized_docs = [self.tokenize(doc) for doc in documents]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_docs, k1=self.k1, b=self.b)
        
        self.is_built = True
        print(f"Built BM25 index with {len(documents)} documents")
        print(f"  Average document length: {self.bm25.avgdl:.1f} tokens")
    
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for top-k documents matching query
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List of (doc_id, score) tuples sorted by relevance
        """
        if not self.is_built:
            raise RuntimeError("Index not built. Call build() first.")
        
        # Tokenize query
        query_tokens = self.tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        # Build results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                results.append((self.doc_ids[idx], float(scores[idx])))
        
        return results
    
    def get_document(self, doc_id: str) -> str:
        """Get document by ID"""
        try:
            idx = self.doc_ids.index(doc_id)
            return self.documents[idx]
        except ValueError:
            return None
    
    def get_statistics(self) -> Dict:
        """Get index statistics"""
        if not self.is_built:
            return {'is_built': False}
        
        doc_lengths = [len(doc) for doc in self.tokenized_docs]
        
        return {
            'is_built': True,
            'num_documents': len(self.documents),
            'avg_doc_length': self.bm25.avgdl,
            'min_doc_length': min(doc_lengths),
            'max_doc_length': max(doc_lengths),
            'vocabulary_size': len(self.bm25.idf),
            'k1': self.k1,
            'b': self.b
        }


def demo():
    """
    Demonstrate BM25 prototype with sample documents
    """
    print("=" * 60)
    print("BM25 PROTOTYPE DEMO")
    print("=" * 60)
    
    # Sample financial documents
    documents = [
        "Apple stock shows strong momentum with rising volume and bullish trend",
        "Bitcoin volatility increases amid regulatory concerns and market uncertainty",
        "NVIDIA quarterly earnings beat expectations with record GPU sales",
        "Federal Reserve signals potential interest rate cuts in upcoming meetings",
        "Microsoft Azure cloud revenue grows 30 percent year over year",
        "Tesla deliveries miss analyst expectations causing stock decline",
        "Gold prices surge as investors seek safe haven assets",
        "Amazon Web Services maintains cloud market leadership position",
        "S&P 500 reaches new all-time high on strong earnings reports",
        "Cryptocurrency market shows signs of recovery after recent selloff"
    ]
    
    doc_ids = [f"doc_{i}" for i in range(len(documents))]
    
    # Build index
    print("\n1. Building BM25 Index...")
    index = BM25Index(k1=1.5, b=0.75)
    index.build(documents, doc_ids)
    
    # Print statistics
    print("\n2. Index Statistics:")
    stats = index.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test queries
    queries = [
        "stock momentum bullish",
        "cryptocurrency bitcoin volatility",
        "cloud computing revenue growth",
        "interest rate federal reserve"
    ]
    
    print("\n3. Search Results:")
    print("-" * 60)
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = index.search(query, k=3)
        
        for rank, (doc_id, score) in enumerate(results, 1):
            doc = index.get_document(doc_id)
            print(f"  {rank}. [{doc_id}] Score: {score:.3f}")
            print(f"     {doc[:60]}...")
    
    print("\n" + "=" * 60)
    print("BM25 Prototype Demo Complete!")
    print("=" * 60)
    
    return index


if __name__ == "__main__":
    demo()

