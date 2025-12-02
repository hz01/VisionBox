"""
Service for module operations
"""
from typing import Dict, List
from core.module_registry import ModuleRegistry


class ModuleService:
    """Service for module operations"""
    
    def __init__(self, module_registry: ModuleRegistry):
        self.module_registry = module_registry
    
    def get_all_modules(self) -> Dict[str, any]:
        """Get all registered modules
        
        Returns:
            Dict with modules and categories
        """
        modules = self.module_registry.list_all()
        categories = self.module_registry.list_by_category()
        
        # Create category index
        category_index = {}
        for cat, module_ids in self.module_registry._categories.items():
            category_index[cat] = [mid for mid in module_ids]
        
        return {
            "modules": modules,
            "categories": category_index
        }
    
    def get_module(self, module_id: str) -> Dict:
        """Get a specific module by ID
        
        Args:
            module_id: Module identifier
            
        Returns:
            Module schema
            
        Raises:
            ValueError: If module not found
        """
        module = self.module_registry.get(module_id)
        return module.get_schema()
    
    def get_modules_by_category(self, category: str) -> List[Dict]:
        """Get modules by category
        
        Args:
            category: Category name
            
        Returns:
            List of module schemas
        """
        result = self.module_registry.list_by_category(category)
        return result.get(category, [])

