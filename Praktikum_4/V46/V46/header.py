#Ein Danke geht an ChatGPT
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp

def format_value(value):
    if hasattr(value, 's'):
        return f"${value:.1u}$"
    return str(value)

def save_latex_table_to_file(arr, header, caption=None, label=None, columnstyle = "0"):
    """
    Save a 2D NumPy array as a LaTeX table to a .tex file with the specified label.

    Args:
        arr (np.ndarray): The NumPy array to convert.
        caption (str): Optional caption for the table.
        label (str): Optional label for referencing the table and as the file name.
        display_bottom_half (bool): True to display the bottom half to the right of the top half.
    """
    if not label:
        raise ValueError("Label must be provided for file naming.")

    table = numpy_array_to_latex_table(arr, header, caption, label, columnstyle)

    with open(f"./build/{label}.tex", "w") as file:
        file.write(table)

def numpy_array_to_latex_table(arr, header='Generischer Header', caption=None, label=None, columnstyle="0"):
    """
    Convert a 2D NumPy array into LaTeX table code.

    Args:
        arr (np.ndarray): The NumPy array to convert.
        caption (str): Optional caption for the table.
        label (str): Optional label for referencing the table.
        display_bottom_half (bool): True to display the bottom half to the right of the top half.

    Returns:
        str: LaTeX code for the table.
    """
    num_cols = len(arr)
    num_rows = len(arr[0])


    table = "\\begin{table}\n"
    table += "\\centering\n"

    if caption:
        table += f"\\caption{{{caption}}}\n"

    if label:
        label = 'tab:'+label 
        table += f"\\label{{{label}}}\n"

    if columnstyle == '0':
        table += "\\begin{tabular}{" + "c " * num_cols + "}\n"
    else:
        table += "\\begin{tabular}"
        table += columnstyle+"\n"

    table += "\\toprule\n"
    table += header + '\\\\\n'
    table += "\\midrule \n"
    
    #arr[col, row]
    for i in range(num_rows):
        formatted_row = ''
        for j in range(num_cols):
            formatted_row += f'{arr[j][i]} &'

#       formatted_row = " & ".join(map(format_value, arr[i, :]))
        
        formatted_row = formatted_row[:-1]
        table += f"{formatted_row} \\\\\n"
        
    table = table.replace('+/-', r' \pm ')
    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"
    table += "\\end{table}\n"
    return table

# Example usage:
#if __name__ == "__main__":
#    data = np.array([[1.23, 2.34, 3.45], [ufloat(4.56, 0.01), 5.67, 6.78], [7.89, 8.90, ufloat(9.01, 0.02)], [10, 11, 12]])
#    label = "my_table"
#    save_latex_table_to_file(data, caption="Example Table", label=label)
#    print(f"LaTeX table saved as {label}.tex")
#


def uzeros(shape):
    """
    This function takes an tuple and returns an array with the shape of the given values
    """
    arr = np.array([[ufloat(0.0, 0.0) for _ in range(shape[1])] for _ in range(shape[0])])
    return arr
