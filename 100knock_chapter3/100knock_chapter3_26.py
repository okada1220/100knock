import json
import re
import pprint

with open('jawiki-country.json', 'r', encoding='utf-8') as f:
    for file in f:
        json_file = json.loads(file)
        if json_file['title'] == 'イギリス':
            json_text = json_file['text']

basic_compile = re.compile("{{基礎情報(.|\\n)*\\n}}?")
basic_search = basic_compile.search(json_text)

basicinfo_compile = re.compile("\|(\w+)\s\s?=\s?((.|\\n\*)*)")
basicinfo_findall = basicinfo_compile.findall(basic_search.group())

dic_basicinfo = {}
for info in basicinfo_findall:
    # 強調除去
    sub_info = re.sub('(\'){2,4}', '', info[1])

    dic_basicinfo[info[0]] = sub_info
pprint.pprint(dic_basicinfo, sort_dicts=False)