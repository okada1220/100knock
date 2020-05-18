import MeCab
import re
import pprint
import pydotplus

def main():
    dot_file_name = 'tree.dot'
    graph = pydotplus.graph_from_dot_file(dot_file_name)
    graph.write_png("iris.png")

if __name__ == '__main__':
    main()