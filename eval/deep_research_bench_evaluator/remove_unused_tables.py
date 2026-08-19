import re
import sys


def find_tables(markdown):
    """Find all tables in the markdown and their positions."""
    lines = markdown.split('\n')
    heading_pattern = re.compile(r'^\*\*\s*Table\s+(\d+)\s*\*\*', re.IGNORECASE)
    tables = []
    
    idx = 0
    while idx < len(lines):
        match = heading_pattern.match(lines[idx].strip())
        if not match:
            idx += 1
            continue
        
        number = match.group(1)
        start_line = idx
        end_line = idx
        cursor = idx + 1
        
        # Find the end of the table block
        while cursor < len(lines):
            trimmed = lines[cursor].strip()
            
            # Empty line ends the table
            if trimmed == '':
                end_line = cursor
                break
            
            # Another table heading ends this table
            if heading_pattern.match(trimmed):
                end_line = cursor - 1
                break
            
            # A markdown heading ends the table
            if trimmed.startswith('#'):
                end_line = cursor - 1
                break
            
            # Table content continues (pipe-separated rows, or "click here" links)
            if trimmed.startswith('|') or is_likely_table_row(trimmed) or trimmed.lower().startswith('[click here'):
                end_line = cursor
                cursor += 1
                continue
            
            # Any other content ends the table
            end_line = cursor - 1
            break
        
        if cursor >= len(lines):
            end_line = len(lines) - 1
        
        end_line = max(end_line, start_line)
        tables.append({
            'number': number,
            'start_line': start_line,
            'end_line': end_line
        })
        idx = end_line + 1
    
    return tables


def is_likely_table_row(line):
    """Check if a line looks like a table row."""
    # Simple heuristic: contains multiple pipe characters
    return line.count('|') >= 2


def is_table_referenced(markdown, table_info, all_tables):
    """Check if a table is referenced elsewhere in the document."""
    lines = markdown.split('\n')
    
    # Get the text before and after the table
    before_lines = lines[:table_info['start_line']]
    after_lines = lines[table_info['end_line'] + 1:]
    
    # Remove other table blocks from the search text to avoid false positives
    text_to_search = '\n'.join(before_lines) + '\n' + '\n'.join(after_lines)
    
    # Look for references like "Table 1", "Table 1.", "Table 1," etc.
    # but not "Table 10" when looking for "Table 1"
    pattern = re.compile(rf'Table\s+{table_info["number"]}(?!\d)', re.IGNORECASE)
    
    return bool(pattern.search(text_to_search))


def remove_unused_tables(input_text):
    """Remove tables that are not referenced elsewhere in the document."""
    tables = find_tables(input_text)
    
    # Determine which tables are unused
    unused_tables = []
    for table in tables:
        if not is_table_referenced(input_text, table, tables):
            unused_tables.append(table)
    
    if not unused_tables:
        return input_text, []
    
    # Remove unused tables (process in reverse order to preserve line numbers)
    lines = input_text.split('\n')
    removed_numbers = []
    
    for table in sorted(unused_tables, key=lambda t: t['start_line'], reverse=True):
        # Remove the table lines
        del lines[table['start_line']:table['end_line'] + 1]
        removed_numbers.append(table['number'])
        
        # Also remove any trailing empty lines that were left behind
        while table['start_line'] < len(lines) and lines[table['start_line']].strip() == '':
            del lines[table['start_line']]
    
    return '\n'.join(lines), list(reversed(removed_numbers))


def main():
    if len(sys.argv) < 2:
        print("Usage: python remove_unused_tables.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.txt', '_cleaned.txt')
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    cleaned_content, removed = remove_unused_tables(content)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)
    
    if removed:
        print(f"Removed {len(removed)} unused table(s): Table {', Table '.join(removed)}")
    else:
        print("No unused tables found.")
    
    print(f"Output written to: {output_file}")


if __name__ == '__main__':
    main()





