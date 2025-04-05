import os
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
import torch
from transformers import AutoTokenizer, AutoModel
import re
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import faiss
import pickle

class IndexedVectorStore:
    """
    A vector store using FAISS for efficient similarity search.
    Supports memory consolidation and deduplication.
    """
    def __init__(self, storage_path: str = "./memory/vector_store"):
        self.storage_path = storage_path
        self.index_path = f"{storage_path}.index"
        self.data_path = f"{storage_path}.pkl"

        # Document storage
        self.documents = []
        self.metadata = []
        self.doc_hashes = set()  # For deduplication

        # Create storage directory if it doesn't exist
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)

        # Initialize FAISS index
        self.embedding_dim = 384  # Default dimension, will be updated when first embedding is added
        self.index = None

        # Load existing data if available
        self.load()

    def _create_index(self, dim):
        """Create a new FAISS index with the specified dimension"""
        self.embedding_dim = dim
        self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine on normalized vectors)

    def _compute_hash(self, text: str) -> str:
        """Compute a hash for text to detect duplicates"""
        # Normalize text for consistent hashing
        normalized = ' '.join(text.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()

    def add(self, text: str, embedding: np.ndarray, metadata: Dict[str, Any] = None):
        """Add a document to the vector store with deduplication"""
        # Create document hash for deduplication
        doc_hash = self._compute_hash(text)

        # Skip if it's a duplicate
        if doc_hash in self.doc_hashes:
            return False

        # First document defines the embedding dimension
        if self.index is None:
            self._create_index(embedding.shape[0])

        # Normalize embedding for cosine similarity
        normalized_embedding = embedding / np.linalg.norm(embedding)

        # Add to FAISS index
        self.index.add(np.array([normalized_embedding], dtype=np.float32))

        # Add to document storage
        self.documents.append(text)
        self.metadata.append(metadata or {})
        self.doc_hashes.add(doc_hash)

        # Save updated data
        self.save()
        return True

    def search(self, query_embedding: np.ndarray, top_k: int = 10, min_similarity: float = 0.25) -> List[Dict[str, Any]]:
      """Search for similar documents using FAISS index with lower threshold"""
      if self.index is None or self.index.ntotal == 0:
          return []

      # Normalize query embedding
      normalized_query = query_embedding / np.linalg.norm(query_embedding)
      normalized_query = np.array([normalized_query], dtype=np.float32)

      # Search the index - get more results than needed to allow for filtering
      # Get at least 2x results to allow for post-filtering
      search_k = min(top_k * 3, self.index.ntotal)
      similarities, indices = self.index.search(normalized_query, search_k)

      results = []
      for i, idx in enumerate(indices[0]):
          if idx != -1 and similarities[0][i] > min_similarity:
              # Include full metadata and document content
              results.append({
                  'text': self.documents[idx],
                  'similarity': float(similarities[0][i]),
                  'metadata': self.metadata[idx],
                  'index': idx
              })

      return results

    def consolidate_memories(self, similarity_threshold: float = 0.85, min_cluster_size: int = 2):
      """a
      Consolidate similar memories to reduce redundancy.
      Uses a simplified approach that doesn't depend on faiss.reconstruct.

      Args:
          similarity_threshold: Threshold for considering documents similar
          min_cluster_size: Minimum size for a cluster to be considered

      Returns:
          Boolean indicating whether any changes were made
      """
      # Skip if no index or not enough documents
      if self.index is None or self.index.ntotal < min_cluster_size:
          return False

      print(f"Consolidating memories... ({self.index.ntotal} documents)")

      try:
          # Create a simple document similarity matrix based on document content
          # This avoids using FAISS's reconstruct function
          doc_count = len(self.documents)

          # Find duplicate or very similar documents by content hash
          content_groups = {}
          for idx, doc in enumerate(self.documents):
              # Create a simpler hash for content comparison
              # This is less precise than vector similarity but should work as a fallback
              content_hash = self._compute_hash(doc[:100])  # Just use first 100 chars
              if content_hash not in content_groups:
                  content_groups[content_hash] = []
              content_groups[content_hash].append(idx)

          # Find groups with multiple documents (potential duplicates)
          duplicate_groups = [indices for content_hash, indices in content_groups.items()
                            if len(indices) >= min_cluster_size]

          if not duplicate_groups:
              print("No duplicate groups found. Nothing to consolidate.")
              return False

          print(f"Found {len(duplicate_groups)} groups of similar documents")

          # Track which indices to keep (start with all)
          indices_to_keep = set(range(doc_count))

          # Process each group
          for group in duplicate_groups:
              # Keep the first document in each group, remove others
              keep_idx = group[0]
              for remove_idx in group[1:]:
                  # Merge metadata from removed document to kept document
                  kept_meta = self.metadata[keep_idx]
                  removed_meta = self.metadata[remove_idx]

                  # Update sources list
                  if 'sources' not in kept_meta:
                      kept_meta['sources'] = []

                  # Add source query if available
                  if 'source_query' in removed_meta and removed_meta['source_query'] not in kept_meta.get('sources', []):
                      kept_meta.setdefault('sources', []).append(removed_meta['source_query'])

                  # Add other sources if available
                  if 'sources' in removed_meta:
                      for source in removed_meta['sources']:
                          if source not in kept_meta['sources']:
                              kept_meta['sources'].append(source)

                  # Remove the duplicate from indices to keep
                  if remove_idx in indices_to_keep:
                      indices_to_keep.remove(remove_idx)

          # Check if we found any duplicates to remove
          if len(indices_to_keep) < doc_count:
              # Create new collections with only kept documents
              new_docs = [self.documents[i] for i in sorted(indices_to_keep)]
              new_metadata = [self.metadata[i] for i in sorted(indices_to_keep)]
              new_hashes = {self._compute_hash(doc) for doc in new_docs}

              # We need to rebuild the index from scratch since we can't extract vectors
              # This is inefficient but necessary given the limitations
              print(f"Rebuilding index with {len(new_docs)} documents (removed {doc_count - len(new_docs)})")

              # Generate new embeddings for all kept documents
              # NOTE: This assumes we have access to an embedding function
              # If this is not available, this approach won't work

              # Create a temporary replacement in-memory index
              if hasattr(self, 'generate_embedding'):
                  # If the class has a generate_embedding method, use it
                  new_embeddings = np.zeros((len(new_docs), self.embedding_dim), dtype=np.float32)
                  for i, doc in enumerate(new_docs):
                      # This assumes generate_embedding is available
                      embedding = self.generate_embedding(doc)
                      new_embeddings[i] = embedding / np.linalg.norm(embedding)  # Normalize

                  # Create new index
                  new_index = faiss.IndexFlatIP(self.embedding_dim)
                  if len(new_embeddings) > 0:
                      new_index.add(new_embeddings)

                  # Update instance variables
                  self.index = new_index
                  self.documents = new_docs
                  self.metadata = new_metadata
                  self.doc_hashes = new_hashes

                  # Save changes
                  self.save()
                  return True
              else:
                  print("Cannot rebuild index - no embedding function available")
                  return False

          return False

      except Exception as e:
          print(f"Error during memory consolidation: {e}")
          return False
    # faiss version if you have faiss.reconstruct
    # def consolidate_memories(self, similarity_threshold: float = 0.85, min_cluster_size: int = 2):
    #     """
    #     Consolidate similar memories to reduce redundancy.
    #     Uses DBSCAN clustering to identify similar documents.
    #     """
    #     if self.index is None or self.index.ntotal < min_cluster_size:
    #         return  # Not enough documents to consolidate

    #     # Get all embeddings
    #     embeddings = np.zeros((self.index.ntotal, self.embedding_dim), dtype=np.float32)
    #     for i in range(self.index.ntotal):
    #         faiss.reconstruct(self.index, i, embeddings[i])

    #     # Compute pairwise similarities
    #     similarities = cosine_similarity(embeddings)

    #     # Use DBSCAN for clustering
    #     eps = 1.0 - similarity_threshold  # Convert similarity to distance
    #     clustering = DBSCAN(eps=eps, min_samples=min_cluster_size, metric='precomputed')
    #     distances = 1.0 - similarities  # Convert similarities to distances
    #     cluster_labels = clustering.fit_predict(distances)

    #     # Process each cluster
    #     unique_clusters = set(cluster_labels)
    #     if -1 in unique_clusters:  # Remove noise cluster
    #         unique_clusters.remove(-1)

    #     if not unique_clusters:
    #         return  # No clusters found

    #     consolidated_docs = []
    #     consolidated_metadata = []
    #     consolidated_hashes = set()
    #     consolidated_embeddings = []

    #     # Keep track of which indices to keep
    #     indices_to_keep = set(range(self.index.ntotal))

    #     for cluster in unique_clusters:
    #         cluster_indices = np.where(cluster_labels == cluster)[0]
    #         if len(cluster_indices) < min_cluster_size:
    #             continue

    #         # Find the centroid document (most representative)
    #         centroid_idx = cluster_indices[0]
    #         for idx in cluster_indices[1:]:
    #             # Mark non-centroid documents for removal
    #             if idx in indices_to_keep:
    #                 indices_to_keep.remove(idx)

    #             # Merge metadata from removed documents to the centroid
    #             centroid_meta = self.metadata[centroid_idx]
    #             removed_meta = self.metadata[idx]

    #             # Combine sources
    #             if 'sources' not in centroid_meta:
    #                 centroid_meta['sources'] = []

    #             if 'source_query' in removed_meta and removed_meta['source_query'] not in centroid_meta.get('sources', []):
    #                 centroid_meta.setdefault('sources', []).append(removed_meta['source_query'])

    #             if 'sources' in removed_meta:
    #                 for source in removed_meta['sources']:
    #                     if source not in centroid_meta['sources']:
    #                         centroid_meta['sources'].append(source)

    #     # Rebuild the index with only the kept documents
    #     kept_indices = sorted(list(indices_to_keep))
    #     if len(kept_indices) < self.index.ntotal:
    #         new_docs = [self.documents[i] for i in kept_indices]
    #         new_metadata = [self.metadata[i] for i in kept_indices]
    #         new_hashes = {self._compute_hash(doc) for doc in new_docs}

    #         # Extract kept embeddings
    #         new_embeddings = np.zeros((len(kept_indices), self.embedding_dim), dtype=np.float32)
    #         for i, orig_idx in enumerate(kept_indices):
    #             faiss.reconstruct(self.index, orig_idx, new_embeddings[i])

    #         # Create new index
    #         new_index = faiss.IndexFlatIP(self.embedding_dim)
    #         new_index.add(new_embeddings)

    #         # Update instance variables
    #         self.index = new_index
    #         self.documents = new_docs
    #         self.metadata = new_metadata
    #         self.doc_hashes = new_hashes

    #         # Save changes
    #         self.save()
    #         return True

    #     return False

    def save(self):
        """Save vector store to disk"""
        # Save index
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)

        # Save documents and metadata
        data = {
            'documents': self.documents,
            'metadata': self.metadata,
            'doc_hashes': list(self.doc_hashes),
            'embedding_dim': self.embedding_dim,
            'updated_at': datetime.now().isoformat()
        }

        with open(self.data_path, 'wb') as f:
            pickle.dump(data, f)

    def load(self):
        """Load vector store from disk"""
        # Check if both files exist
        if not (os.path.exists(self.index_path) and os.path.exists(self.data_path)):
            return

        try:
            # Load documents and metadata
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)

            self.documents = data.get('documents', [])
            self.metadata = data.get('metadata', [])
            self.doc_hashes = set(data.get('doc_hashes', []))
            self.embedding_dim = data.get('embedding_dim', 384)

            # Load index if there are documents
            if self.documents:
                self.index = faiss.read_index(self.index_path)
        except Exception as e:
            print(f"Error loading vector store: {e}")
            # Initialize new store if loading fails
            self.documents = []
            self.metadata = []
            self.doc_hashes = set()
            self.index = None


class EnhancedMemoryManager:
    """
    Manages long-term memory for LLM conversations with advanced features:
    - FAISS indexing for efficient retrieval
    - Memory consolidation and deduplication
    - Automatic memorization with toggle control
    """
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        memory_dir: str = "./memory",
        device: str = None,
        auto_memorize: bool = True
    ):
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)

        self.user_stores = {}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.auto_memorize = auto_memorize

        # For very small models, we can load the embedding model
        # For production, would use a smaller model or API
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.embedding_dim = self.model.config.hidden_size
            print(f"Loaded embedding model: {model_name}")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            # Fallback to random embeddings for demo
            self.tokenizer = None
            self.model = None
            self.embedding_dim = 384  # Common dimension

        # Schedule for periodic maintenance
        self.last_consolidation = datetime.now()
        self.consolidation_interval = 60 * 60  # 1 hour in seconds

    def _get_user_store(self, user_id: str) -> IndexedVectorStore:
        """Get or create a vector store for a user"""
        if user_id not in self.user_stores:
            store_path = os.path.join(self.memory_dir, f"{user_id}_store")
            self.user_stores[user_id] = IndexedVectorStore(store_path)
        return self.user_stores[user_id]

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate an embedding for a text snippet"""
        if self.model is None:
            # Fallback to random embeddings for demo
            return np.random.random(self.embedding_dim)

        # Tokenize and embed the text
        inputs = self.tokenizer(text, return_tensors="pt",
                               truncation=True, max_length=512,
                               padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use mean pooling to get a single vector
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
        return embedding

    def extract_key_information(self, query: str, response: str) -> List[str]:
        """Extract key information from conversation to remember"""
        # In a production system, you'd use an LLM to extract key information
        # For this implementation, we'll use a more sophisticated rule-based approach

        # Extract factual statements (sentences with potential facts)
        combined_text = f"{query} {response}"
        sentences = re.split(r'[.!?]', combined_text)
        key_info = []

        # Process each sentence with lower thresholds
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # LOWER THRESHOLD: was 15, now 10
                continue

            words = sentence.lower().split()
            if not words:
                continue

            # SIMPLIFY CRITERIA: Almost any non-trivial sentence gets stored
            contains_entity = bool(re.search(r'\b[A-Z][a-z]+\b', sentence))
            contains_numbers = bool(re.search(r'\b\d+\b', sentence))

            # MUCH LESS RESTRICTIVE: Either has capitalized words or numbers or is long enough
            if contains_entity or contains_numbers or len(words) > 5:
                clean_sentence = re.sub(r'\s+', ' ', sentence)
                key_info.append(clean_sentence)

        # INCREASE NUMBER OF MEMORIES STORED
        return key_info[:10]  # Return up to 10 memories instead of 5

        ## limits
        # # Simple heuristics to identify potential facts
        # fact_starters = [
        #     "is", "was", "are", "were", "has", "have", "had",
        #     "can", "will", "should", "must", "might", "may",
        #     "the", "a", "an", "in", "born", "created", "developed",
        #     "contains", "consists", "includes"
        # ]

        # entity_patterns = [
        #     r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Proper names (e.g., John Smith)
        #     r'\b[A-Z][a-z]+\b',  # Capitalized words (potential named entities)
        #     r'\b\d{4}\b',  # Years
        #     r'\b\d+\.\d+\b',  # Numbers with decimals (potential statistics)
        #     r'\b\d+%\b',  # Percentages
        # ]

        # for sentence in sentences:
        #     sentence = sentence.strip()
        #     if len(sentence) < 15:  # Skip very short sentences
        #         continue

        #     words = sentence.lower().split()
        #     if not words:
        #         continue

        #     # Check if it starts with a fact starter
        #     is_potential_fact = words[0] in fact_starters

        #     # Check if it contains entities (using regex patterns)
        #     contains_entity = False
        #     for pattern in entity_patterns:
        #         if re.search(pattern, sentence):
        #             contains_entity = True
        #             break

        #     # Check if it contains numbers (often indicates factual info)
        #     contains_numbers = bool(re.search(r'\b\d+\b', sentence))

        #     # Keywords that suggest this is explaining something
        #     explanation_keywords = ["because", "therefore", "means", "defined as", "refers to"]
        #     is_explanation = any(keyword in sentence.lower() for keyword in explanation_keywords)

        #     # Add if it meets our criteria for being a fact
        #     if (is_potential_fact and (contains_entity or contains_numbers)) or is_explanation:
        #         # Clean up the sentence
        #         clean_sentence = re.sub(r'\s+', ' ', sentence)
        #         key_info.append(clean_sentence)

        # # Limit the number of extracted facts (prioritize ones with entities)
        # return key_info[:5]

    def add_memory(self, user_id: str, query: str, response: str) -> int:
      """Add a new memory with better correction detection"""
      if not self.auto_memorize:
          return 0

      # Check if maintenance is due
      self._check_maintenance(user_id)

      # Get user's vector store
      store = self._get_user_store(user_id)

      # ENHANCED CORRECTION DETECTION
      is_correction = False
      correction_indicators = ["incorrect", "wrong", "not true", "false", "mistake",
                              "that's not", "but that's", "but that is", "actually",
                              "in fact", "not correct"]

      # Check if the query contains correction indicators
      query_lower = query.lower()
      if any(indicator in query_lower for indicator in correction_indicators):
          is_correction = True
          print(f"[Memory] Detected correction in: '{query}'")

      # Extract key information
      key_info = self.extract_key_information(query, response)

      # Store each key piece of information
      memories_added = 0
      for info in key_info:
          embedding = self.generate_embedding(info)

          # Add metadata about whether this is a correction
          memory_type = "correction" if is_correction else "auto_extract"

          # For corrections, add additional keywords to improve retrieval
          if is_correction:
              # Extract key terms from the correction
              terms = set(query_lower.split())
              # Filter out common words
              stop_words = {'a', 'an', 'the', 'but', 'and', 'or', 'is', 'are', 'was', 'were', 'to', 'for', 'in', 'on', 'at', 'by'}
              keywords = ' '.join([term for term in terms if term not in stop_words and len(term) > 2])

              # Store with added metadata
              metadata = {
                  'source_query': query,
                  'source_response': response[:100] + "..." if len(response) > 100 else response,
                  'timestamp': datetime.now().isoformat(),
                  'type': memory_type,
                  'is_correction': True,
                  'correction_keywords': keywords,
                  'priority': 10  # Higher priority for corrections
              }
          else:
              metadata = {
                  'source_query': query,
                  'source_response': response[:100] + "..." if len(response) > 100 else response,
                  'timestamp': datetime.now().isoformat(),
                  'type': memory_type,
                  'priority': 1  # Normal priority
              }

          added = store.add(
              text=info,
              embedding=embedding,
              metadata=metadata
          )
          if added:
              memories_added += 1

      return memories_added

    # i liked this version more
    # def add_memory(self, user_id: str, query: str, response: str) -> int:
    #     """
    #     Add a new memory from a conversation
    #     Returns number of memories added
    #     """
    #     if not self.auto_memorize:
    #         return 0

    #     # Check if maintenance is due
    #     self._check_maintenance(user_id)

    #     # Get user's vector store
    #     store = self._get_user_store(user_id)

    #     # Extract key information
    #     key_info = self.extract_key_information(query, response)

    #     # Store each key piece of information
    #     memories_added = 0
    #     for info in key_info:
    #         embedding = self.generate_embedding(info)
    #         added = store.add(
    #             text=info,
    #             embedding=embedding,
    #             metadata={
    #                 'source_query': query,
    #                 'source_response': response[:100] + "..." if len(response) > 100 else response,
    #                 'timestamp': datetime.now().isoformat(),
    #                 'type': 'auto_extract'
    #             }
    #         )
    #         if added:
    #             memories_added += 1

    #     return memories_added


    def retrieve_relevant_memories(self, user_id: str, query: str, top_k: int = 8) -> str:
      """
      Enhanced memory retrieval with better relevance sorting and contextual understanding.
      """
      # Get user's vector store
      store = self._get_user_store(user_id)

      # Generate embedding for query
      query_embedding = self.generate_embedding(query)

      # Get more initial results with lower threshold for post-processing
      results = store.search(query_embedding, top_k=top_k*3, min_similarity=0.2)

      # Detect specific query types
      is_math_query = bool(re.search(r'\d+\s*[\+\-\*\/]\s*\d+', query))
      is_transliteration_query = bool(re.search(r'transliterate|script|alphabet|letter', query.lower()))
      is_ordering_query = bool(re.search(r'(first|last|next|previous|order|sequence|position|number|place)\s+(letter|word|item)', query.lower()))

      # Boost memories based on query type
      for result in results:
          result['boost'] = 0

          # Initial score based on similarity
          result['final_score'] = result['similarity']

          # Boost corrections significantly
          if result.get('metadata', {}).get('is_correction', False):
              result['boost'] += 0.4
              result['final_score'] += 0.4

          # Apply specific domain boosts
          if is_math_query and re.search(r'\d+\s*[\+\-\*\/]\s*\d+', result['text']):
              result['boost'] += 0.3

          if is_transliteration_query and re.search(r'script|alphabet|letter|transliterate', result['text'].lower()):
              result['boost'] += 0.25

          if is_ordering_query and re.search(r'(first|last|next|previous|position|number|order)', result['text'].lower()):
              result['boost'] += 0.2

          # Boost recently added memories
          timestamp = result.get('metadata', {}).get('timestamp', '')
          if timestamp:
              try:
                  memory_time = datetime.fromisoformat(timestamp)
                  # Boost memories from last 10 minutes
                  if (datetime.now() - memory_time).total_seconds() < 600:
                      result['boost'] += 0.15
                      result['final_score'] += 0.15
              except:
                  pass

      # Sort by final score
      sorted_results = sorted(results, key=lambda x: x.get('final_score', 0), reverse=True)

      # Take top results after boosting
      top_results = sorted_results[:top_k]

      # Group results into clear categories
      corrections = []
      factual_info = []
      general_info = []

      for result in top_results:
          if result.get('metadata', {}).get('is_correction', False):
              corrections.append(result)
          elif any(term in result['text'].lower() for term in ['definition', 'fact', 'rule', 'alphabet', 'order']):
              factual_info.append(result)
          else:
              general_info.append(result)

      # Compose the memory text with clear sections
      memory_text = ""

      if corrections:
          memory_text += "IMPORTANT CORRECTIONS (You MUST apply these):\n"
          for i, result in enumerate(corrections):
              memory_text += f"- {result['text']}\n"
          memory_text += "\n"

      if factual_info:
          memory_text += "FACTUAL INFORMATION:\n"
          for i, result in enumerate(factual_info):
              memory_text += f"- {result['text']}\n"
          memory_text += "\n"

      if general_info and (not corrections or not factual_info):
          memory_text += "OTHER RELEVANT INFORMATION:\n"
          for i, result in enumerate(general_info):
              memory_text += f"- {result['text']}\n"

      return memory_text

    def save_conversation(self, user_id: str, conversation: List[Dict[str, Any]]) -> int:
        """
        Save an entire conversation with enhanced memory extraction
        Returns number of memories added
        """
        if not self.auto_memorize:
            return 0

        if len(conversation) < 2:
            return 0  # Skip empty conversations

        # Check if maintenance is due
        self._check_maintenance(user_id)

        # Pair user queries with assistant responses
        memories_added = 0
        for i in range(1, len(conversation), 2):
            if i + 1 < len(conversation):
                query = conversation[i].get('content', '')
                response = conversation[i+1].get('content', '')

                if query and response:
                    memories_added += self.add_memory(user_id, query, response)

        return memories_added

    def _check_maintenance(self, user_id: str):
        """Check if memory maintenance (consolidation) is due"""
        now = datetime.now()
        time_diff = (now - self.last_consolidation).total_seconds()

        if time_diff > self.consolidation_interval:
            self.consolidate_memories(user_id)
            self.last_consolidation = now

    def consolidate_memories(self, user_id: str) -> bool:
        """Consolidate memories for a user to reduce redundancy"""
        store = self._get_user_store(user_id)
        return store.consolidate_memories()

    def toggle_auto_memorize(self) -> bool:
        """Toggle automatic memorization on/off"""
        self.auto_memorize = not self.auto_memorize
        return self.auto_memorize

    def get_stats(self, user_id: str) -> Dict[str, Any]:
        """Get memory statistics for a user"""
        store = self._get_user_store(user_id)
        return {
            'total_memories': len(store.documents),
            'auto_memorize': self.auto_memorize,
            'last_consolidation': self.last_consolidation.isoformat(),
        }