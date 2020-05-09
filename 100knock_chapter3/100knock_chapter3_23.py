import json
import re

with open('jawiki-country.json', 'r', encoding='utf-8') as f:
    for file in f:
        json_file = json.loads(file)
        if json_file['title'] == 'イギリス':
            json_text = json_file['text']

json_line = json_text.split()

for i in range(len(json_line)):
    for n in range(2, 5):
        session_compile = re.compile("^={" + str(n) + "}(\s?\w+\s?)={" + str(n) + "}")
        session_search = session_compile.search(json_line[i])
        if bool(session_search):
            print(session_search.group(), n-1)