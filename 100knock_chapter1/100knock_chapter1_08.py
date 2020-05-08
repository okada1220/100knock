def cipher(text):
    changed_text = ''
    for i in range(len(text)):
        if text[i].islower() == True:
            changed_text += chr(219-ord(text[i]))
        else:
            changed_text += text[i]
    return changed_text

print('textを入力してください。')
text = input()
changed_text = cipher(text)
print('暗号化します。')
print(changed_text)
print('復号化します。')
original_text = cipher(changed_text)
print(original_text)