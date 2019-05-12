class Node(object):
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinaryTree(object):
    def __init__(self, root):
        self.root = Node(root)

    def search(self, find_val):
        return self.preorder_search(tree.root,find_val)

    def print_tree(self):
        return self.postorder_print(tree.root," ")

    def preorder_search(self, start, find_val):
        if start:
            if start.value == find_val:
                return True
            else:
                return self.preorder_search(start.right,find_val) or self.preorder_search(start.left,find_val) 
        return False

    def preorder_print(self, start,traversal):
        if start:
            traversal =(str(start.value)+ " ")
            traversal = self.preorder_print(start.left,traversal) 
            traversal = self.preorder_print(start.right,traversal)
        return traversal
    
    def postorder_print(self, start,traversal):
        if start:
            traversal = self.postorder_print(start.left,traversal) 
            traversal = self.postorder_print(start.right,traversal)
            traversal += (str(start.value)+ " ")
        return traversal   
    
    def inorder_print(self, start,traversal):
        if start:
            traversal = self.inorder_print(start.left,traversal)
            traversal += str(start.value) + " " 
            traversal = self.inorder_print(start.right,traversal)
        return traversal
        
