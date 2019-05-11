"""Write a HashTable class that stores strings
in a hash table, where keys are calculated
using the first two letters of the string."""

class HashTable(object):
    def __init__(self):
        self.table = [None]*10000

    def store(self, string):
        index = self.calculate_hash_value(string)
        if index!=-1:
            if (self.table[index] == None):
              self.table[index] = [string]
            else:
              self.table[index].append(string)    
        

    def lookup(self, string):
        index = self.calculate_hash_value(string)
        if index!=-1:
             if self.table[index]!= None:
                 if string in self.table[index]:
                        return index
        
        return -1

    def calculate_hash_value(self, string):
        value = (ord(string[0]) * 100) + ord(string[1])
        return value 
    
