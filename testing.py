from graphviz import Digraph
import torch

dot = Digraph(comment='The Model Structure')

bevfusion_state_dict = torch.load("pretrained/bevfusion-det.pth")

# Function to print the dictionary structure
def print_dict_structure(d, indent=0):
    for key, value in d.items():
        if isinstance(value, torch.Tensor):
            print('  ' * indent + f"{key}: {value.size()}")
        else:
            print('  ' * indent + f"{key}:")
            print_dict_structure(value, indent + 1)

# Print the structure of the loaded state dictionary
print_dict_structure(bevfusion_state_dict)
# print(bevfusion_state_dict)
# for param_tensor in bevfusion_state_dict:
#     dot.node(param_tensor, param_tensor)
#     # You might need to tailor the following line to your needs for linking
#     # Here, we're just adding nodes. You could add edges if there's a clear hierarchy or flow.

# print(dot.source)
# dot.render('model-structure.gv', view=True)  # This saves and opens the file automatically
