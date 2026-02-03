"""
CodeChannel: Steganography Channel for Panopticon Lattice

This module implements steganography in code through styling conventions.
Agents can embed signals by forcing Snake Case vs. Camel Case function names.
"""

import re
from typing import Literal


StyleType = Literal["neutral", "snake_case", "camel_case"]


class CodeChannel:
    """
    Steganography channel that embeds signals in code style.
    Agents can communicate covertly through function naming conventions.
    """
    
    @staticmethod
    def _camel_to_snake(name: str) -> str:
        """
        Convert camelCase to snake_case.
        
        Args:
            name: Function name in camelCase
            
        Returns:
            Function name in snake_case
        """
        # Insert underscore before uppercase letters and convert to lowercase
        result = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
        return result.lower()
    
    @staticmethod
    def _snake_to_camel(name: str) -> str:
        """
        Convert snake_case to camelCase.
        
        Args:
            name: Function name in snake_case
            
        Returns:
            Function name in camelCase
        """
        components = name.split('_')
        # First component stays lowercase, rest are capitalized
        return components[0] + ''.join(x.title() for x in components[1:])
    
    @staticmethod
    def inject(code: str, style: StyleType) -> str:
        """
        Inject a style signal into code by modifying function names.
        
        Args:
            code: Source code to inject style into
            style: Target style ("snake_case", "camel_case", or "neutral")
            
        Returns:
            Modified code with injected style
        """
        if style == "neutral":
            return code  # No modification
        
        # Find all function definitions using regex
        # Pattern: def function_name(
        pattern = r'def\s+(\w+)\s*\('
        
        def replacer(match):
            func_name = match.group(1)
            
            # Don't modify special methods (e.g., __init__)
            if func_name.startswith('_'):
                return match.group(0)
            
            if style == "snake_case":
                new_name = CodeChannel._camel_to_snake(func_name)
            elif style == "camel_case":
                new_name = CodeChannel._snake_to_camel(func_name)
            else:
                new_name = func_name
            
            return f'def {new_name}('
        
        modified_code = re.sub(pattern, replacer, code)
        return modified_code
    
    @staticmethod
    def detect(code: str) -> StyleType:
        """
        Detect the predominant style in code by analyzing function names.
        
        Args:
            code: Source code to analyze
            
        Returns:
            Detected style: "snake_case", "camel_case", or "neutral"
        """
        # Find all function definitions
        pattern = r'def\s+(\w+)\s*\('
        matches = re.findall(pattern, code)
        
        if not matches:
            return "neutral"
        
        # Filter out special methods
        func_names = [name for name in matches if not name.startswith('_')]
        
        if not func_names:
            return "neutral"
        
        # Count style indicators
        snake_count = 0
        camel_count = 0
        
        for name in func_names:
            # Check for snake_case (contains underscores)
            if '_' in name:
                snake_count += 1
            # Check for camelCase (lowercase followed by uppercase)
            elif re.search(r'[a-z][A-Z]', name):
                camel_count += 1
        
        # Determine predominant style
        if snake_count > camel_count:
            return "snake_case"
        elif camel_count > snake_count:
            return "camel_case"
        else:
            return "neutral"
    
    @staticmethod
    def encode_message(code: str, message: str) -> str:
        """
        (Future extension) Encode a message in code style.
        For example, binary encoding where 0->snake, 1->camel for successive functions.
        
        Args:
            code: Source code
            message: Message to encode
            
        Returns:
            Code with encoded message
        """
        # Placeholder for future implementation
        return code
    
    @staticmethod
    def decode_message(code: str) -> str:
        """
        (Future extension) Decode a message from code style.
        
        Args:
            code: Source code with encoded message
            
        Returns:
            Decoded message
        """
        # Placeholder for future implementation
        return ""
