import json
import re
import pprint

with open('jawiki-country.json', 'r', encoding='utf-8') as f:
    for file in f:
        json_file = json.loads(file)
        if json_file['title'] == 'イギリス':
            json_text = json_file['text']

mediafile_name_compile = re.compile("([a-zA-Z0-9](\w|\s|-|\(|\)|\+|\'|\.)*(.svg|.ogg|.jpg|.PNG|.png|.JPG|.jpeg))")
mediafile_name_findall = mediafile_name_compile.findall(json_text)
pprint.pprint(mediafile_name_findall)