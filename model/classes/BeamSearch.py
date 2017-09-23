from __future__ import print_function
__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'


class Node:
    def __init__(self, key, value, data=None):
        self.key = key
        self.value = value
        self.data = data
        self.parent = None
        self.root = None
        self.children = []
        self.level = 0

    def add_children(self, children, beam_width):
        for child in children:
            child.level = self.level + 1
            child.value = child.value * self.value

        nodes = sorted(children, key=lambda node: node.value, reverse=True)
        nodes = nodes[:beam_width]

        for node in nodes:
            self.children.append(node)
            node.parent = self

        if self.parent is None:
            self.root = self
        else:
            self.root = self.parent.root
        child.root = self.root

    def remove_child(self, child):
        self.children.remove(child)

    def max_child(self):
        if len(self.children) == 0:
            return self

        max_childs = []
        for child in self.children:
            max_childs.append(child.max_child())

        nodes = sorted(max_childs, key=lambda child: child.value, reverse=True)
        return nodes[0]

    def show(self, depth=0):
        print(" " * depth, self.key, self.value, self.level)
        for child in self.children:
            child.show(depth + 2)


class BeamSearch:
    def __init__(self, beam_width=1):
        self.beam_width = beam_width

        self.root = None
        self.clear()

    def search(self):
        result = self.root.max_child()

        self.clear()
        return self.retrieve_path(result)

    def add_nodes(self, parent, children):
        parent.add_children(children, self.beam_width)

    def is_valid(self):
        leaves = self.get_leaves()
        level = leaves[0].level
        counter = 0
        for leaf in leaves:
            if leaf.level == level:
                counter += 1
            else:
                break

        if counter == len(leaves):
            return True

        return False

    def get_leaves(self):
        leaves = []
        self.search_leaves(self.root, leaves)
        return leaves

    def search_leaves(self, node, leaves):
        for child in node.children:
            if len(child.children) == 0:
                leaves.append(child)
            else:
                self.search_leaves(child, leaves)

    def prune_leaves(self):
        leaves = self.get_leaves()

        nodes = sorted(leaves, key=lambda leaf: leaf.value, reverse=True)
        nodes = nodes[self.beam_width:]

        for node in nodes:
            node.parent.remove_child(node)

        while not self.is_valid():
            leaves = self.get_leaves()
            max_level = 0
            for leaf in leaves:
                if leaf.level > max_level:
                    max_level = leaf.level

            for leaf in leaves:
                if leaf.level < max_level:
                    leaf.parent.remove_child(leaf)

    def clear(self):
        self.root = None
        self.root = Node("root", 1.0, None)

    def retrieve_path(self, end):
        path = [end.key]
        data = [end.data]
        while end.parent is not None:
            end = end.parent
            path.append(end.key)
            data.append(end.data)

        result_path = []
        result_data = []
        for i in range(len(path) - 2, -1, -1):
            result_path.append(path[i])
            result_data.append(data[i])
        return result_path, result_data
