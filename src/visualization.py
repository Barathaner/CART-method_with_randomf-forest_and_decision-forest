
from graphviz import Digraph
import random
def visualize_tree(node, dot=None, parent_id=None, edge_label="", node_id=0):
    if dot is None:
        dot = Digraph()
        dot.attr('node', shape='box')  # Set the shape of nodes to boxes
        initial_label = f'"{node_id}: {node.label if node.is_leaf_node() else node.condition}"'
        dot.node(str(node_id), initial_label)
        parent_id = node_id

    if node.is_leaf_node():
        leaf_label = f'"Leaf {node_id}: {node.label}"'
        dot.node(str(node_id), leaf_label)
        if parent_id is not None:
            dot.edge(str(parent_id), str(node_id), label=edge_label)
    else:
        # Create a label for the condition
        if node.condition.is_numeric:
            condition_label = f"{node.condition.feature} < {node.condition.value}"
        else:
            # For categorical features, the split is binary but visualized as one node
            condition_label = f"{node.condition.feature} == {node.condition.value}"

        node_label = f'"{node_id}: {condition_label}"'
        dot.node(str(node_id), node_label)
        if parent_id is not None:
            dot.edge(str(parent_id), str(node_id), label=edge_label)

        # Recursively visualize the subtree
        # We define "Yes" for the condition being true, and "No" for false
        if node.left:
            left_id = random.randint(0, 1000000)
            visualize_tree(node.left, dot, node_id, "Yes", left_id)
        if node.right:
            right_id = random.randint(0, 1000000)
            visualize_tree(node.right, dot, node_id, "No", right_id)

    return dot


