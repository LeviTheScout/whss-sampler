import re

with open('poster.tex', 'r') as f:
    content = f.read()

# Replace column minipages with fixed height
content = content.replace(r'\begin{minipage}[t]{\pcolwidth}', r'\begin{minipage}[t][85cm]{\pcolwidth}')

# Insert \vfill between blocks within the same column
# A block ends with \end{block} and the next starts with \begin{block}
# We only want to add \vfill if it's between blocks (not across columns)

def replacer(match):
    return r'\end{block}' + '\n\\vfill\n' + r'\begin{block}'

# Find all occurrences of \end{block} \n\n \begin{block}
content = re.sub(r'\\end\{block\}\s*\\begin\{block\}', replacer, content)

with open('poster.tex', 'w') as f:
    f.write(content)
