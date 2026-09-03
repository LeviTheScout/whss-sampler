import re

with open('poster.tex', 'r') as f:
    content = f.read()

# Remove \vfill
content = re.sub(r'\\end\{block\}\s*\\vfill\s*\\begin\{block\}', r'\\end{block}\n\n\\begin{block}', content)

# Remove fixed height
content = content.replace(r'\begin{minipage}[t][85cm]{\pcolwidth}', r'\begin{minipage}[t]{\pcolwidth}')

with open('poster.tex', 'w') as f:
    f.write(content)
