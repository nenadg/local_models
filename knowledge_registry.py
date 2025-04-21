"""
Knowledge registry for TinyLlama Chat system.
Manages registration, loading and unloading of knowledge domains.
"""

import os
import json
import threading
import shutil
from typing import List, Dict, Any, Optional, Set, Callable
from datetime import datetime

class KnowledgeRegistry:
    """
    Central registry for managing knowledge domains.
    
    The KnowledgeRegistry orchestrates the registration, discovery, loading,
    and unloading of knowledge domains. It provides a centralized interface
    for domain management and cross-domain operations.
    """
    
    def __init__(
        self,
        base_directory: str = "./knowledge_domains",
        embedding_function: Optional[Callable] = None,
        embedding_dim: int = 384,
        max_active_domains: int = 5,
        enable_fractal: bool = True,
        max_fractal_levels: int = 3
    ):
        """
        Initialize the knowledge registry.
        
        Args:
            base_directory: Base directory for domain storage
            embedding_function: Function to generate embeddings
            embedding_dim: Dimension of embeddings
            max_active_domains: Maximum number of domains to keep active in memory
            enable_fractal: Whether to enable fractal embeddings
            max_fractal_levels: Maximum number of fractal levels
        """
        self.base_directory = base_directory
        self.embedding_function = embedding_function
        self.embedding_dim = embedding_dim
        self.max_active_domains = max_active_domains
        self.enable_fractal = enable_fractal
        self.max_fractal_levels = max_fractal_levels
        
        # Create base directory if it doesn't exist
        os.makedirs(base_directory, exist_ok=True)
        
        # Registry file paths
        self.registry_path = os.path.join(base_directory, "registry.json")
        
        # Thread lock for concurrent access
        self._lock = threading.RLock()
        
        # Initialize registry
        self.domains = {}  # domain_id -> domain instance
        self.active_domains = {}  # domain_id -> last accessed time
        self.registry_metadata = {
            "total_domains": 0,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        
        # Load registry if it exists
        self._load_registry()
    
    def _load_registry(self) -> bool:
        """
        Load registry metadata from file.
        
        Returns:
            Success boolean
        """
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    registry_data = json.load(f)
                
                # Update registry metadata
                self.registry_metadata = registry_data.get("metadata", self.registry_metadata)
                
                # Get domain list, but don't load domains yet
                domain_list = registry_data.get("domains", [])
                
                # Track how many domains we have
                self.registry_metadata["total_domains"] = len(domain_list)
                
                return True
            except Exception as e:
                print(f"Error loading registry: {e}")
        
        return False
    
    def _save_registry(self) -> bool:
        """
        Save registry metadata to file.
        
        Returns:
            Success boolean
        """
        with self._lock:
            try:
                # Update timestamp
                self.registry_metadata["last_updated"] = datetime.now().isoformat()
                
                # Create registry data structure
                registry_data = {
                    "metadata": self.registry_metadata,
                    "domains": []
                }
                
                # Add domain information
                for domain_id, domain in self.domains.items():
                    if hasattr(domain, 'metadata'):
                        registry_data["domains"].append({
                            "domain_id": domain_id,
                            "name": domain.metadata.get("name", domain_id),
                            "description": domain.metadata.get("description", ""),
                            "stats": domain.metadata.get("stats", {"total_items": 0})
                        })
                
                # Save to file
                with open(self.registry_path, 'w', encoding='utf-8') as f:
                    json.dump(registry_data, f, indent=2)
                
                return True
            except Exception as e:
                print(f"Error saving registry: {e}")
                return False
    
    def register_domain(self, domain) -> bool:
        """
        Register a domain with the registry.
        
        Args:
            domain: Domain instance to register
            
        Returns:
            Success boolean
        """
        with self._lock:
            try:
                domain_id = domain.domain_id
                
                # Add to domains dict
                self.domains[domain_id] = domain
                
                # Update active domains
                self.active_domains[domain_id] = datetime.now().timestamp()
                
                # Update registry metadata
                self.registry_metadata["total_domains"] = len(self.domains)
                
                # Save registry
                self._save_registry()
                
                # Manage active domains
                self._manage_active_domains()
                
                return True
            except Exception as e:
                print(f"Error registering domain: {e}")
                return False
    
    def unregister_domain(self, domain_id: str) -> bool:
        """
        Unregister a domain from the registry.
        
        Args:
            domain_id: Domain ID to unregister
            
        Returns:
            Success boolean
        """
        with self._lock:
            if domain_id not in self.domains:
                print(f"Domain {domain_id} not found in registry")
                return False
            
            try:
                # Remove from domains dict
                domain = self.domains.pop(domain_id)
                
                # Remove from active domains
                if domain_id in self.active_domains:
                    del self.active_domains[domain_id]
                
                # Update registry metadata
                self.registry_metadata["total_domains"] = len(self.domains)
                
                # Save registry
                self._save_registry()
                
                return True
            except Exception as e:
                print(f"Error unregistering domain: {e}")
                return False
    
    def get_domain(self, domain_id: str, load_if_needed: bool = True):
        """
        Get a domain by ID.
        
        Args:
            domain_id: Domain ID to get
            load_if_needed: Whether to load domain if not already active
            
        Returns:
            Domain instance or None if not found
        """
        with self._lock:
            # Check if domain is already active
            if domain_id in self.domains:
                # Update access time
                self.active_domains[domain_id] = datetime.now().timestamp()
                return self.domains[domain_id]
            
            # If not active and we're allowed to load
            if load_if_needed:
                return self._load_domain(domain_id)
            
            return None
    
    def _load_domain(self, domain_id: str):
        """
        Load a domain from disk.
        
        Args:
            domain_id: Domain ID to load
            
        Returns:
            Domain instance or None if not found
        """
        domain_directory = os.path.join(self.base_directory, domain_id)
        
        if not os.path.exists(domain_directory):
            print(f"Domain directory not found: {domain_directory}")
            return None
        
        try:
            # Load domain metadata
            metadata_path = os.path.join(domain_directory, "metadata.json")
            if not os.path.exists(metadata_path):
                print(f"Domain metadata not found: {metadata_path}")
                return None
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Import KnowledgeDomain class
            try:
                from knowledge_domain import KnowledgeDomain
            except ImportError:
                print("Error importing KnowledgeDomain class")
                return None
            
            # Create domain instance
            domain = KnowledgeDomain(
                name=metadata.get("name", domain_id),
                description=metadata.get("description", ""),
                base_directory=self.base_directory,
                embedding_function=self.embedding_function,
                embedding_dim=self.embedding_dim,
                enable_fractal=self.enable_fractal,
                max_fractal_levels=self.max_fractal_levels
            )
            
            # Register domain
            self.register_domain(domain)
            
            return domain
            
        except Exception as e:
            print(f"Error loading domain {domain_id}: {e}")
            return None
    
    def _manage_active_domains(self):
        """Manage active domains to stay within memory limits."""
        with self._lock:
            # If we're under the limit, nothing to do
            if len(self.active_domains) <= self.max_active_domains:
                return
            
            # Sort domains by last accessed time
            sorted_domains = sorted(
                self.active_domains.items(),
                key=lambda x: x[1]  # sort by timestamp
            )
            
            # Unload oldest domains until we're within limits
            domains_to_unload = sorted_domains[:len(sorted_domains) - self.max_active_domains]
            
            for domain_id, _ in domains_to_unload:
                # Save domain before unloading
                if domain_id in self.domains:
                    domain = self.domains[domain_id]
                    
                    # Save domain state
                    if hasattr(domain, 'save'):
                        domain.save()
                    
                    # Remove from active domains and domains dict
                    del self.active_domains[domain_id]
                    del self.domains[domain_id]
    
    def create_domain(self, name: str, description: str = "") -> Optional[str]:
        """
        Create a new knowledge domain.
        
        Args:
            name: Domain name
            description: Domain description
            
        Returns:
            Domain ID if successful, None otherwise
        """
        try:
            # Import KnowledgeDomain class
            from knowledge_domain import KnowledgeDomain

            # Remove any leading/trailing colons or whitespace from the name
            name = name.strip().strip(':')

            # Create domain instance
            domain = KnowledgeDomain(
                name=name,  # Clean name
                description=description,
                base_directory=self.base_directory,
                embedding_function=self.embedding_function,
                embedding_dim=self.embedding_dim,
                enable_fractal=self.enable_fractal,
                max_fractal_levels=self.max_fractal_levels
            )
            
            # Register domain
            success = self.register_domain(domain)
            
            return domain.domain_id if success else None
            
        except Exception as e:
            print(f"Error creating domain: {e}")
            return None
    
    def delete_domain(self, domain_id: str) -> bool:
        """
        Delete a knowledge domain completely.
        
        Args:
            domain_id: Domain ID to delete
            
        Returns:
            Success boolean
        """
        with self._lock:
            # Get domain
            domain = self.get_domain(domain_id)
            
            if not domain:
                print(f"Domain {domain_id} not found")
                return False
            
            try:
                # Call domain's delete method if available
                if hasattr(domain, 'delete'):
                    success = domain.delete()
                else:
                    # Fallback to manual deletion
                    domain_directory = os.path.join(self.base_directory, domain_id)
                    if os.path.exists(domain_directory):
                        shutil.rmtree(domain_directory)
                    success = True
                
                # Unregister domain
                self.unregister_domain(domain_id)
                
                return success
            except Exception as e:
                print(f"Error deleting domain {domain_id}: {e}")
                return False
    
    def list_domains(self) -> List[Dict[str, Any]]:
        """
        List all registered domains.
        
        Returns:
            List of domain information dictionaries
        """
        # First, scan directory for domains that might not be in registry
        self._discover_domains()
        
        # Load registry to get current information
        try:
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                registry_data = json.load(f)
            
            return registry_data.get("domains", [])
        except Exception as e:
            print(f"Error listing domains: {e}")
            
            # Fallback to active domains
            return [
                {
                    "domain_id": domain_id,
                    "name": domain.metadata.get("name", domain_id) if hasattr(domain, 'metadata') else domain_id,
                    "description": domain.metadata.get("description", "") if hasattr(domain, 'metadata') else "",
                    "stats": domain.metadata.get("stats", {"total_items": 0}) if hasattr(domain, 'metadata') else {"total_items": 0}
                }
                for domain_id, domain in self.domains.items()
            ]
    
    def _discover_domains(self) -> int:
        """
        Scan directory for domains not in registry.
        
        Returns:
            Number of new domains discovered
        """
        # Skip if base directory doesn't exist
        if not os.path.exists(self.base_directory):
            return 0
        
        discovered = 0
        
        # Find all subdirectories that might be domains
        for item in os.listdir(self.base_directory):
            domain_directory = os.path.join(self.base_directory, item)
            
            # Skip if not a directory or already in registry
            if not os.path.isdir(domain_directory) or item in self.domains:
                continue
            
            # Check if it has metadata.json (indicates a domain)
            metadata_path = os.path.join(domain_directory, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    # Basic validation of metadata
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # Check if it has required fields
                    if "name" in metadata and "domain_id" in metadata:
                        # Add to registry metadata (but don't load domain)
                        registry_entry = {
                            "domain_id": metadata["domain_id"],
                            "name": metadata["name"],
                            "description": metadata.get("description", ""),
                            "stats": metadata.get("stats", {"total_items": 0})
                        }
                        
                        # Add to registry data
                        with open(self.registry_path, 'r', encoding='utf-8') as f:
                            registry_data = json.load(f)
                        
                        registry_data["domains"].append(registry_entry)
                        
                        with open(self.registry_path, 'w', encoding='utf-8') as f:
                            json.dump(registry_data, f, indent=2)
                        
                        discovered += 1
                except Exception as e:
                    print(f"Error discovering domain {item}: {e}")
        
        return discovered
    
    def search_across_domains(self, query: str, domain_ids: List[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for knowledge across multiple domains.
        
        Args:
            query: Search query
            domain_ids: List of domain IDs to search (None for all active domains)
            top_k: Maximum number of results per domain
            
        Returns:
            List of search results from all domains
        """
        all_results = []
        
        # Determine which domains to search
        domains_to_search = []
        if domain_ids:
            # Search specific domains
            for domain_id in domain_ids:
                domain = self.get_domain(domain_id)
                if domain:
                    domains_to_search.append(domain)
        else:
            # Search all active domains
            domains_to_search = list(self.domains.values())
        
        # Search each domain
        for domain in domains_to_search:
            if hasattr(domain, 'search'):
                try:
                    domain_results = domain.search(query, top_k=top_k)
                    
                    # Add domain info to results
                    for result in domain_results:
                        result['domain_id'] = domain.domain_id
                        result['domain_name'] = domain.metadata.get("name", domain.domain_id) if hasattr(domain, 'metadata') else domain.domain_id
                    
                    all_results.extend(domain_results)
                except Exception as e:
                    print(f"Error searching domain {domain.domain_id}: {e}")
        
        # Sort results by similarity
        all_results.sort(
            key=lambda x: x.get('search_metadata', {}).get('similarity', 0),
            reverse=True
        )
        
        return all_results[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary of registry statistics
        """
        stats = {
            "total_domains": self.registry_metadata.get("total_domains", 0),
            "active_domains": len(self.active_domains),
            "last_updated": self.registry_metadata.get("last_updated", ""),
            "domains": {}
        }
        
        # Add domain stats
        for domain_id, domain in self.domains.items():
            if hasattr(domain, 'get_stats'):
                try:
                    domain_stats = domain.get_stats()
                    stats["domains"][domain_id] = domain_stats
                except Exception as e:
                    print(f"Error getting stats for domain {domain_id}: {e}")
        
        return stats
    
    def cleanup(self):
        """Clean up resources used by the registry."""
        with self._lock:
            # Save all domains
            for domain_id, domain in self.domains.items():
                if hasattr(domain, 'save'):
                    try:
                        domain.save()
                    except Exception as e:
                        print(f"Error saving domain {domain_id} during cleanup: {e}")
            
            # Save registry
            self._save_registry()
            
            # Clear dictionaries
            self.domains.clear()
            self.active_domains.clear()