import torch



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
