import sys
import copy
import pickle
import numpy as np
import re
from index import InvertIndex
from index import decode

class Term():
    def __init__(self, indexes, length):
        self.indexes = indexes
        if indexes is None:
            self.indexes = []
        self.current_position = -1
        self.length = length
        self.len_ind = len(self.indexes)

    def Evaluate(self):
        self.current_position += 1
        if self.current_position < self.len_ind:
            return self.indexes[self.current_position]
        else:
            return self.length
    
    def FastSearch(self, val):
        mid = (self.current_position + self.len_ind) // 2
        low = self.current_position
        high = self.len_ind - 1
        while (self.indexes[mid] != val and low <= high):
            if val > self.indexes[mid]:
                low = mid + 1
            else:
                high = mid - 1
            mid = (low + high) // 2
        if low > high:
            return low
        return mid
    
    def GoTo(self, ind, intersect = False):
        if self.current_position < self.len_ind:
            if not intersect:
                while (self.current_position < self.len_ind and \
                    self.indexes[self.current_position] < ind):
                    self.current_position += 1
                self.current_position -= 1
            else:
                self.current_position = self.FastSearch(ind) - 1
        else:
            return self.length
    
class Not():
    def __init__(self, length):
        self.node = None
        self.current = -2
        self.result = -1
        self.length = length
        
    def Evaluate(self):
        self.current += 1
        if self.current >= self.length:
            return self.length
        
        while self.current == self.result:
            self.result = self.node.Evaluate()
            self.current += 1
        return self.current
        
    def GoTo(self, ind, intersect = False):
        self.node.GoTo(ind, intersect)
        self.result = self.node.Evaluate()       
        self.current = ind - 1
    
class Or():
    def __init__(self, length):
        self.node = [None, None]
        self.result = np.array([-1, -1], dtype = int)
        self.length = length
        
    def Evaluate(self):
        if self.result[0] != self.result[1]:
            _min = np.argmin(self.result)
            self.result[_min] = self.node[_min].Evaluate()
        else:
            if self.result[0] == self.length:
                return self.length
            self.result[0] = self.node[0].Evaluate()
            self.result[1] = self.node[1].Evaluate()
        return np.min(self.result)
        
    def GoTo(self, ind, intersect = False):
        self.node[0].GoTo(ind, intersect)
        self.node[1].GoTo(ind, intersect)
        self.result[0] = ind
        self.result[1] = ind
    
class And():
    def __init__(self, length):
        self.node = [None, None]
        self.result = np.array([-1, -1])
        self.length = length
        
    def Evaluate(self):
        self.result[0] = self.node[0].Evaluate()
        self.result[1] = self.node[1].Evaluate()
        while (self.result[0] != self.result[1]):
            _min = np.argmin(self.result)
            _max = 1 - _min
            self.node[_min].GoTo(self.result[_max], True) # find element in less 
            self.result[_min] = self.node[_min].Evaluate()
        return np.min(self.result)

    def GoTo(self, ind, intersect = False):
        self.node[0].GoTo(ind, intersect)
        self.node[1].GoTo(ind, intersect)
        self.result[0] = ind
        self.result[1] = ind

class QTree():  
    def __init__(self, index):
        self.index = index
        self.length = len(self.index.doc_url)

    def parse_el(self, element):
        if (type(element) == list):
            return 3
        elif (element == '|'):
            return 0
        elif (element == '&'):
            return 1
        elif (element == '!'):
            return 2
        return 4

    def get_priority(self, command):
        return [self.parse_el(el) for el in command]

    def MakeTree(self, command):
        priorities = np.array(self.get_priority(command), dtype = int)
        min_pos = np.argmin(priorities)
        if (priorities[min_pos] == 4): #term, return list of nums
            if type(command) == list:
                return Term(decode(self.index.index[command[0]]), self.length)
            else:
                return Term(decode(self.index.index[command]), self.length)
        if (priorities[min_pos] == 3): #list
            return self.MakeTree(command[0])
        if (priorities[min_pos] == 2):
            result = Not(self.length)
            result.node = self.MakeTree(command[1])
            return result
        if (priorities[min_pos] == 1):
            result = And(self.length)
        if (priorities[min_pos] == 0):
            result = Or(self.length)
        result.node[0] = self.MakeTree(command[0 : min_pos])
        result.node[1] = self.MakeTree(command[min_pos + 1 : None])
        return result
    
    def search(self, command, original):
        self.tree = self.MakeTree(command)
        output = ""
        count_results = 0
        result = []
        next_id = self.tree.Evaluate()
        while next_id < self.length:
            result.append(self.index.doc_url[next_id])
            output += self.index.doc_url[next_id] + '\n'
            next_id = self.tree.Evaluate()
            count_results += 1
        output = original + str(count_results) + '\n' + output
        return result, output

class Parser():
    def __init__(self, command):
        self.command = command
    
    def DeleteExtra(self, tokens):
        while (len(tokens) == 1 and type(tokens[0]) == list):
            tokens = tokens[0]
        return tokens

    def BracketsToLists(self, tokens):
        lvl = 0
        start = -1
        pos = 0

        while (pos < len(tokens)):
            if tokens[pos] == '(':
                if lvl == 0:
                    start = pos
                lvl += 1
            if tokens[pos] == ')':
                lvl -= 1
                if lvl == 0:
                    new = self.BracketsToLists(copy.copy(tokens[start + 1 : pos]))
                    new = self.DeleteExtra(new)
                    del tokens[start : pos + 1]
                    tokens.insert(start, new)
                    pos = start
            pos += 1
        tokens = self.DeleteExtra(tokens)
        if (type(tokens) != list):
            tokens = list(tokens)
        return tokens

    def Parse(self):
        pre_tokens = re.findall(r'\w+|[\(\)&\|!]', self.command)
        tokens = [s.lower() for s in pre_tokens]
        return self.BracketsToLists(tokens)

def main():
    total_output = ''
    with open('storage.pkl', 'rb') as f:
        index = pickle.load(f)
    search_tree = QTree(index)
    for query in sys.stdin:
        if query == '\n':
            break
        parsed_commands = Parser(query).Parse()
        result, output = search_tree.search(parsed_commands, query)
        total_output += output
    print(total_output)

if __name__ == "__main__":
    main()
