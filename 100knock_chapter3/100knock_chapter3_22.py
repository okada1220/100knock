import json
import re

with open('jawiki-country.json', 'r', encoding='utf-8') as f:
    for file in f:
        json_file = json.loads(file)
        if json_file['title'] == 'イギリス':
            json_text = json_file['text']

category_compile = re.compile("\[\[Category:(.*)\]\]")
category_findall = category_compile.findall(json_text)
print(category_findall)