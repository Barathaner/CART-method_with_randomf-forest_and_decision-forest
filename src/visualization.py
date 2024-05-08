from graphviz import Digraph
import random

def visualize_tree(node, dot=None, parent_id=None, edge_label="", node_id=0):
    """
    Visualizes a decision tree using Graphviz.
    
    Args:
        node (Node): The current node in the decision tree to visualize. The node should have
                     attributes `is_leaf_node`, `label`, `condition`, `left`, and `right`.
        dot (Digraph, optional): The Graphviz Digraph object. If None, a new Digraph is created.
        parent_id (int, optional): The ID of the parent node in the graph to link with current node.
        edge_label (str, optional): The label for the edge connecting the parent node to the current node.
        node_id (int, optional): The unique identifier for the current node. Default is 0.
        
    Returns:
        Digraph: A Graphviz Digraph object representing the decision tree graphically.
        
    Description:
        This function recursively visualizes a decision tree by creating nodes and edges in a Graphviz
        Digraph. Nodes are shaped as boxes, with leaf nodes labeled with their class labels and decision
        nodes labeled with their split conditions. The function handles both numeric splits (less than)
        and categorical splits (equals). Each recursive call processes the left and right children of the
        current node, with edges labeled 'Yes' for left (condition met) and 'No' for right (condition not met).
    """
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
            left_id = random.randint(0, 1000000)  # Generate a unique ID for the left child
            visualize_tree(node.left, dot, node_id, "Yes", left_id)
        if node.right:
            right_id = random.randint(0, 1000000)  # Generate a unique ID for the right child
            visualize_tree(node.right, dot, node_id, "No", right_id)

    return dot
