def main():
    semantic_analogy = {'sum':0, 'corrent':0}
    syntactic_analogy = {'sum':0, 'corrent':0}
    with open('questions-words-result.txt', 'r', encoding='utf-8') as file:
        for line in file:
            words = line.split()
            if words[0] == ':':
                if 'gram' in words[1]:
                    flag = 1
                else:
                    flag = 0
            else:
                if flag == 0:
                    semantic_analogy['sum'] += 1
                    if words[3] == words[4]:
                        semantic_analogy['corrent'] += 1
                else:
                    syntactic_analogy['sum'] += 1
                    if words[3] == words[4]:
                        syntactic_analogy['corrent'] += 1

    print(semantic_analogy)
    print(syntactic_analogy)
    semantic_accuracy = semantic_analogy['corrent'] / semantic_analogy['sum']
    syntactic_accuracy = syntactic_analogy['corrent'] / syntactic_analogy['sum']
    print(semantic_accuracy)
    print((syntactic_accuracy))

if __name__ == '__main__':
    main()