import json
import re

with open('jawiki-country.json', 'r', encoding='utf-8') as f:
    for file in f:
        json_file = json.loads(file)
        if json_file['title'] == 'イギリス':
            json_text = json_file['text']

json_lines = json_text.split('\n')

category_compile = re.compile("\[\[Category:.*\]\]")
for json_line in json_lines:
    if bool(category_compile.search(json_line)):
        print(json_line)