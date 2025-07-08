from smolagents import tool
import os

@tool
def output_save_to_file(filename: str, data: str) -> None:
    """
    Saves the final output to a file.
        Note: If the output has some code, store it accrding to the programing language's extension
        along with the original markdown (Eg. calling this tool multiple times for
        fibonacci_theory.md, fibonacci_recursive.py, fibonacci_recursive_dynamic.py, fibonacci_iterative.py)

    Args:
        filename (str): Full Name of the file being saved (Eg. market_research.md)
        data (str): The output to be stored in the file (Regular old UTF-8)

    Returns:
        None

    """
    filepath = "/home/cyto/dev/ctix-analysis/outputs/"  + filename
    if os.path.exists(filepath):
        # Get the base name and extension of the original file
        base_name, extension = os.path.splitext(filepath)

        # Create a new file name with "_1" prepended
        # TODO: make it +1 each time
        new_file_path = f"{base_name}_1{extension}"

        # Write data to the new file
        with open(new_file_path, 'w') as new_file:
            new_file.write(data)

        print(f"New file created: {new_file_path}")
    else:
        with open(filepath, 'w') as f:
            f.write(data)
