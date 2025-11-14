import re

class MarkdownTerminalRenderer:
    def __init__(self):
        # ANSI color codes
        self.colors = {
            'header': '\033[1;36m',      # Bold cyan
            'bold': '\033[1m',           # Bold
            'italic': '\033[3m',         # Italic
            'code': '\033[38;5;208m',    # Orange
            'blockquote': '\033[38;5;240m', # Gray
            'link': '\033[34m',          # Blue
            'reset': '\033[0m',          # Reset
            'hr': '\033[38;5;240m',      # Gray for horizontal rule
            'list': '\033[38;5;33m',     # Blue for list markers
        }
        
        # Compile regex patterns
        self.patterns = {
            'header': re.compile(r'^(#{1,6})\s+(.+)$'),
            'bold': re.compile(r'\*\*(.*?)\*\*'),
            'italic': re.compile(r'\*(.*?)\*'),
            'inline_code': re.compile(r'`([^`]+)`'),
            'link': re.compile(r'\[([^\]]+)\]\(([^)]+)\)'),
            'image': re.compile(r'!\[([^\]]+)\]\(([^)]+)\)'),
            'blockquote': re.compile(r'^>\s*(.+)$'),
            'unordered_list': re.compile(r'^[\*\-\+]\s+(.+)$'),
            'ordered_list': re.compile(r'^\d+\.\s+(.+)$'),
            'horizontal_rule': re.compile(r'^---+\s*$|^___+\s*$|^\*\*\*+\s*$'),
            'code_block': re.compile(r'^```(\w*)$'),
        }
    
    def render(self, markdown_text: str) -> str:
        """Parse and render markdown text for terminal output."""
        lines = markdown_text.split('\n')
        output_lines = []
        in_code_block = False
        code_block_lang = ''
        in_list = False
        list_type = None
        
        for i, line in enumerate(lines):
            # Handle code blocks
            if self.patterns['code_block'].match(line.strip()):
                if not in_code_block:
                    in_code_block = True
                    match = self.patterns['code_block'].match(line.strip())
                    code_block_lang = match.group(1) if match.group(1) else ''
                    output_lines.append(f"{self.colors['code']}```{code_block_lang}{self.colors['reset']}")
                else:
                    in_code_block = False
                    output_lines.append(f"{self.colors['code']}```{self.colors['reset']}")
                continue
            
            if in_code_block:
                output_lines.append(f"{self.colors['code']}{line}{self.colors['reset']}")
                continue
            
            # Skip empty lines
            if not line.strip():
                if in_list:
                    in_list = False
                    list_type = None
                output_lines.append('')
                continue
            
            # Process markdown elements
            rendered_line = self._parse_line(line)
            
            # Handle lists
            if self.patterns['unordered_list'].match(line.strip()) or self.patterns['ordered_list'].match(line.strip()):
                if not in_list:
                    in_list = True
                    list_type = 'unordered' if self.patterns['unordered_list'].match(line.strip()) else 'ordered'
                
                if list_type == 'unordered':
                    marker = f"{self.colors['list']}•{self.colors['reset']}"
                else:
                    match = self.patterns['ordered_list'].match(line.strip())
                    marker = f"{self.colors['list']}{match.group(0).split('.')[0]}.{self.colors['reset']}"
                
                content = self.patterns['unordered_list'].match(line.strip()) or self.patterns['ordered_list'].match(line.strip())
                rendered_line = f"  {marker} {content.group(1)}"
            
            output_lines.append(rendered_line)
        
        return '\n'.join(output_lines)
    
    def _parse_line(self, line: str) -> str:
        """Parse a single line of markdown."""
        # Headers
        header_match = self.patterns['header'].match(line)
        if header_match:
            level = len(header_match.group(1))
            text = header_match.group(2)
            underline = '=' if level == 1 else '-' if level == 2 else ''
            return f"{self.colors['header']}{text}{self.colors['reset']}\n{underline * len(text)}" if underline else f"{self.colors['header']}{text}{self.colors['reset']}"
        
        # Horizontal rule
        if self.patterns['horizontal_rule'].match(line.strip()):
            return f"{self.colors['hr']}{'─' * 60}{self.colors['reset']}"
        
        # Blockquote
        blockquote_match = self.patterns['blockquote'].match(line)
        if blockquote_match:
            return f"{self.colors['blockquote']}│ {blockquote_match.group(1)}{self.colors['reset']}"
        
        # Apply inline formatting
        line = self.patterns['bold'].sub(
            f"{self.colors['bold']}\\1{self.colors['reset']}", line
        )
        line = self.patterns['italic'].sub(
            f"{self.colors['italic']}\\1{self.colors['reset']}", line
        )
        line = self.patterns['inline_code'].sub(
            f"{self.colors['code']}\\1{self.colors['reset']}", line
        )
        line = self.patterns['link'].sub(
            f"{self.colors['link']}\\1 ({self.colors['italic']}\\2{self.colors['link']}){self.colors['reset']}", line
        )
        line = self.patterns['image'].sub(
            f"{self.colors['italic']}[Image: \\1] (\\2){self.colors['reset']}", line
        )
        
        return line

def print_markdown(markdown_text: str):
    """Convenience function to directly print markdown to terminal."""
    renderer = MarkdownTerminalRenderer()
    rendered_text = renderer.render(markdown_text)
    print(rendered_text)
