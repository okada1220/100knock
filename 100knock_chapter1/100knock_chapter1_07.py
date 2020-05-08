def make_script(x, y, z):
    return str(x) + '時の' + str(y) + 'は' + str(z)

print('xは？')
x = input()
print('yは？')
y = input()
print('zは？')
z = input()

script = make_script(x, y, z)
print(script)