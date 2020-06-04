import certifi
import urllib3
from bs4 import BeautifulSoup

def get_html_data(url):
    http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',
                               ca_certs=certifi.where())
    r = http.request('GET', url)

    if r.status != 200:
        raise IOError('http request failure')

    data = r.data.decode()
    return data

def rename_to_true_name(wiki_string):
    if ',' in wiki_string:
        print('find,')
    split_data = wiki_string.split(',')
    true_data = ' '.join(split_data[::-1]).strip()
    return true_data

def main():
    url = 'https://en.wikipedia.org/wiki/List_of_sovereign_states'
    data = get_html_data(url)
    soup = BeautifulSoup(data, 'html5lib')
    table = soup.find('table', attrs={'class':'sortable wikitable'})
    row_list = table.find_all('tr')

    country = []
    for row in row_list:
        print('------------------')
        col = row.find('td')
        if not col:
            print('none item!')
            continue
        elif col['style'] == 'text-align:center;':
            print('skipped!')
            continue

        display_none = row.find('span', attrs = {'style':'display:none'})
        if display_none:
            display_none.extract()

        sup_ref = row.find('sup', attrs = {'class':'reference'})
        if sup_ref:
            sup_ref.extract()

        row_data = col.text.strip()
        print(row_data)

        # 新しく出てきた国名を country に加える
        if '→' in row_data:
            key_value = row_data.split('→')
            name = rename_to_true_name(key_value[0].strip())
            print('name', name)
            if not name in country:
                country.append(name)

        elif '–' in row_data:
            key_value = row_data.split('–')
            name = rename_to_true_name(key_value[0].strip())
            print('name', name)
            if not name in country:
                country.append(name)

        else:
            name = rename_to_true_name(row_data)
            print('name', name)
            if not name in country:
                country.append(name)

    print(country)
    with open('country.txt', 'w', encoding='utf-8') as file:
        for country_name in country:
            file.write(country_name + '\n')

if __name__ == '__main__':
    main()