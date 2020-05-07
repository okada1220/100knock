import json

with open('jawiki-country.json', 'r', encoding='utf-8') as file:
    print(json.load(file))