import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from sklearn.cluster import DBSCAN

class SharpenedVectorStore:
    """Utility class for vector space sharpening operations"""

    def sharpen_embeddings(self, embeddings: np.ndarray, sharpening_factor: float = 0.3,
                          min_cluster_size: int = 3) -> np.ndarray:
        """
        Sharpen embeddings by pushing them closer to their cluster centroids.
        Similar to how image sharpening enhances edges between regions.

        Args:
            embeddings: Array of embedding vectors
            sharpening_factor: How strongly to push vectors toward centroids (0-1)
            min_cluster_size: Minimum size for a cluster to be considered

        Returns:
            Sharpened embedding vectors
        """
        if len(embeddings) < min_cluster_size * 2:
            # Not enough vectors to perform meaningful clustering
            return embeddings

        # Compute similarity matrix
        similarity_matrix = np.zeros((len(embeddings), len(embeddings)))
        for i in range(len(embeddings)):
            for j in range(len(embeddings)):
                # Compute cosine similarity
                similarity_matrix[i, j] = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))

        # Convert similarities to distances (1 - similarity)
        distances = 1 - similarity_matrix

        # Try to import sklearn for clustering
        try:
            from sklearn.cluster import DBSCAN

            # Cluster embeddings using DBSCAN
            clustering = DBSCAN(eps=0.2, min_samples=min_cluster_size, metric='precomputed')
            labels = clustering.fit_predict(distances)

            # Get unique cluster labels (ignoring noise points with label -1)
            unique_clusters = set(label for label in labels if label != -1)

            if not unique_clusters:
                # No valid clusters found
                return embeddings

            # Compute cluster centroids
            centroids = {}
            for cluster in unique_clusters:
                # Get indices of vectors in this cluster
                cluster_indices = np.where(labels == cluster)[0]
                # Calculate centroid
                cluster_vectors = embeddings[cluster_indices]
                centroid = np.mean(cluster_vectors, axis=0)
                # Normalize centroid
                centroid = centroid / np.linalg.norm(centroid)
                centroids[cluster] = centroid

            # Sharpen embeddings by moving them toward their centroids
            sharpened_embeddings = embeddings.copy()

            for i, label in enumerate(labels):
                if label != -1:  # Skip noise points
                    centroid = centroids[label]
                    # Vector pointing from embedding to centroid
                    direction = centroid - embeddings[i]
                    # Move embedding toward centroid by sharpening_factor
                    sharpened_embeddings[i] = embeddings[i] + (direction * sharpening_factor)
                    # Renormalize
                    sharpened_embeddings[i] = sharpened_embeddings[i] / np.linalg.norm(sharpened_embeddings[i])

            return sharpened_embeddings

        except ImportError:
            # Fall back to simpler method if sklearn is not available
            print("Warning: sklearn not available, using simplified sharpening")
            return embeddings

    def update_search_with_sharpening(self, search_method, sharpening_factor=0.3):
        """
        Decorator to enhance search with vector sharpening

        Args:
            search_method: The original search method to enhance
            sharpening_factor: How strongly to sharpen the results (0-1)

        Returns:
            Enhanced search method
        """
        def enhanced_search(query_embedding, top_k=10, min_similarity=0.25, apply_sharpening=True):
            # Get preliminary results
            results = search_method(query_embedding, top_k=top_k*2, min_similarity=min_similarity)

            if not results or not apply_sharpening:
                return results[:top_k]

            # We need to have the embeddings directly
            # Extract embeddings using FAISS when possible or skip sharpening
            try:
                # First, let's try to get embeddings for the results
                # We'll use normalized query for similarity calculations
                normalized_query = query_embedding / np.linalg.norm(query_embedding)

                # Since we don't have direct access to the vectors, we'll use a different approach
                # Instead of sharpening the result vectors, we'll just sharpen the similarity scores
                for result in results:
                    # Use existing similarity score
                    similarity = result['similarity']

                    # Apply a simplified sharpening to the similarity score
                    if similarity > 0.6:  # High similarity scores get boosted
                        sharpened_similarity = similarity + ((similarity - 0.6) * sharpening_factor)
                    else:  # Low similarity scores get reduced
                        sharpened_similarity = similarity * (1.0 - (0.3 * sharpening_factor))

                    # Store original and updated similarities
                    result['original_similarity'] = similarity
                    result['similarity'] = min(1.0, max(0.0, sharpened_similarity))

                # Resort based on new similarity scores
                results.sort(key=lambda x: x['similarity'], reverse=True)

                return results[:top_k]

            except Exception as e:
                # If we encounter any issues, just return the original results
                print(f"Sharpening skipped: {str(e)}")
                return results[:top_k]

        return enhanced_search

