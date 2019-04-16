# -*- coding: UTF-8 -*-

class NodeClass:
    def __init__(self, code, depth):
        '''
            code: (int), node code
            is_leaf_node: (bool), if the node is a leaf node
            depth: the depth of the node in the hierarchy (from 0)
        '''
        self.code = code
        self.parent_code = -1
        self.depth = depth
        self.children_code = set()

    def add_child_code(self, child_code):
        self.children_code.add(child_code)

    def change_parent_code(self, parent_code):
        self.parent_code = parent_code

    def sort_children_code(self):
        self.children_code = list(self.children_code)
        self.children_code.sort()

    def get_parent_code(self):
        return self.parent_code

    def get_children_code(self):
        return self.children_code

    def get_node_depth(self):
        return self.depth