import re
from typing import List, Dict, Optional
from config import BlockRule

class StreamCleaner:
    def __init__(self, block_rules: List[BlockRule], replacements: Dict[str, str]):
        self.block_rules = block_rules
        self.replacements = replacements
        self.active_block: Optional[BlockRule] = None
        self.buffer = ""
    
    def process_chunk(self, chunk: str) -> str:
        self.buffer += chunk
        output = ""
        
        while True:
            if not self.active_block:
                # Find earliest complete start tag among all configured blocks
                best_pos = -1
                best_rule = None
                
                for rule in self.block_rules:
                    pos = self.buffer.find(rule.start_tag)
                    if pos != -1:
                        if best_pos == -1 or pos < best_pos:
                            best_pos = pos
                            best_rule = rule
                
                if best_rule is None:
                    # No COMPLETE tags found. 
                    # Check if the buffer ends with a potential start of a tag.
                    max_partial = 0
                    for rule in self.block_rules:
                        # Check suffixes of the buffer that match prefixes of the start_tag
                        for i in range(min(len(self.buffer), len(rule.start_tag) - 1), 0, -1):
                            if self.buffer.endswith(rule.start_tag[:i]):
                                max_partial = max(max_partial, i)
                                break
                    
                    safe_len = len(self.buffer) - max_partial
                    if safe_len > 0:
                        output += self.buffer[:safe_len]
                        self.buffer = self.buffer[safe_len:]
                    break
                else:
                    # Output text before tag
                    output += self.buffer[:best_pos]
                    # Discard start tag
                    self.buffer = self.buffer[best_pos + len(best_rule.start_tag):]
                    self.active_block = best_rule
                    output += best_rule.replacement
            else:
                # We are inside a block, look for the end tag for this specific block
                end_pos = self.buffer.find(self.active_block.end_tag)
                if end_pos != -1:
                    # Found end tag. Discard everything before it and the tag itself.
                    self.buffer = self.buffer[end_pos + len(self.active_block.end_tag):]
                    self.active_block = None
                else:
                    # Discard everything in the buffer except what might be part of an end tag
                    max_partial = 0
                    # For end tags, we only care about the active block's end tag
                    for i in range(min(len(self.buffer), len(self.active_block.end_tag) - 1), 0, -1):
                        if self.buffer.endswith(self.active_block.end_tag[:i]):
                            max_partial = max(max_partial, i)
                            break
                    
                    safe_len = len(self.buffer) - max_partial
                    self.buffer = self.buffer[safe_len:]
                    break
        
        return self._apply_final_strips(output)
        
    def flush(self) -> str:
        if self.active_block:
            self.buffer = ""
            self.active_block = None
            return ""
        
        out = self.buffer
        self.buffer = ""
        return self._apply_final_strips(out)

    def _apply_final_strips(self, text: str) -> str:
        if not text:
            return text
            
        for find_str, replace_str in self.replacements.items():
            text = text.replace(find_str, replace_str)
        
        # Collapse repeating punctuation
        text = re.sub(r"([.!?,])\1+", r"\1", text)
        return text

    def stop(self):
        self.buffer = ""
        self.active_block = None
