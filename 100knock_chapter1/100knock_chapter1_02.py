script1 = 'パトカー'
script2 = 'タクシー'
chain_script = ''

for i in range(max(len(script1), len(script2))):
    chain_script += script1[i] + script2[i]

print(chain_script)