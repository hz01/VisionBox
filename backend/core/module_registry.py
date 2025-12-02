"""
Module registry for managing CV modules
"""
from typing import Dict, List, Optional
from core.base_module import BaseCVModule


class ModuleRegistry:
    """Central registry for all CV modules"""
    
    def __init__(self):
        self._modules: Dict[str, BaseCVModule] = {}
        self._categories: Dict[str, List[str]] = {}
    
    def register(self, module: BaseCVModule) -> None:
        """Register a CV module
        
        Args:
            module: Module instance to register
            
        Raises:
            ValueError: If module ID already exists
        """
        if module.module_id in self._modules:
            raise ValueError(f"Module '{module.module_id}' is already registered")
        
        self._modules[module.module_id] = module
        
        # Track by category
        category = module.category
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(module.module_id)
    
    def get(self, module_id: str) -> BaseCVModule:
        """Get a module by ID
        
        Args:
            module_id: Module identifier
            
        Returns:
            Module instance
            
        Raises:
            ValueError: If module not found
        """
        if module_id not in self._modules:
            raise ValueError(f"Module '{module_id}' not found")
        return self._modules[module_id]
    
    def list_all(self) -> List[Dict]:
        """List all registered modules with their schemas"""
        return [module.get_schema() for module in self._modules.values()]
    
    def list_by_category(self, category: Optional[str] = None) -> Dict[str, List[Dict]]:
        """List modules grouped by category
        
        Args:
            category: Optional category filter
            
        Returns:
            Dict mapping category names to lists of module schemas
        """
        if category:
            if category not in self._categories:
                return {}
            return {
                category: [
                    self._modules[mid].get_schema() 
                    for mid in self._categories[category]
                ]
            }
        
        result = {}
        for cat, module_ids in self._categories.items():
            result[cat] = [
                self._modules[mid].get_schema() 
                for mid in module_ids
            ]
        return result
    
    def exists(self, module_id: str) -> bool:
        """Check if a module exists"""
        return module_id in self._modules

