class Node(object):
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BST(object):
    def __init__(self, root):
        self.root = Node(root)

    def insert(self, new_val):
        self.in_help(self.root,new_val)

    def search(self, find_val):
        return self.s_help(self.root,find_val)
    
    def s_help(self,current,find_val):
        if current:
            if current.value == find_val:
                return True
            elif current.value < find_val:
                return self.s_help(current.right,find_val)
            else : 
                return self.s_help(current.left,find_val)
        return False        
    
    def in_help(self,current,new_val):
        if current.value < new_val:
           if current.right:
              self.in_help(current.right,new_val) 
           else :
              current.right = Node(new_val)
        else:
           if current.left:
              self.in_help(current.left,new_val) 
           else :
              current.left = Node(new_val)
