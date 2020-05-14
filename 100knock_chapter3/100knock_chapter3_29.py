import json
import re
import pprint
from urllib.request import Request, urlopen
from urllib.parse import urlencode

URL = 'https://en.wikipedia.org/w/api.php?'

def set_params(image_title):
    params = {
        'action': 'query',
        'format': 'json',
        'prop': 'imageinfo',
        'titles': image_title,
        'iiprop': 'url'
    }
    return params

def main():
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
        dic_basicinfo[info[0]] = info[1]

    country_image = dic_basicinfo['国旗画像']
    print(country_image)

    request_url = URL + urlencode(set_params(country_image))
    print(request_url)
    req = Request(request_url)
    with urlopen(req) as res:
        res_json = res.read()
    res_decode = res_json.decode('utf-8')
    wiki = json.loads(res_decode)
    print(res_decode)
    print(wiki)
    print(wiki['query']['pages'])

if __name__ == '__main__':
    main()
