import sys

#
# Author: Kade Peay
#

def calculate_difference_percentage(file1_lines, file2_lines):
    total_lines = max(len(file1_lines), len(file2_lines))
    if total_lines == 0:
        return 0.0  # both files are empty
    
    differences = 0
    
    # Compare line by line up to the length of the shorter file
    min_length = min(len(file1_lines), len(file2_lines))
    for i in range(min_length):
        if file1_lines[i] != file2_lines[i]:
            differences += 1
    
    # Add the extra lines from the longer file
    differences += abs(len(file1_lines) - len(file2_lines))
    
    return (differences / total_lines) * 100

# Read in both files from command line arguments
try: 
    fileName1, fileName2 = sys.argv[1], sys.argv[2]
except Exception as e:
    print(f"Error with arguments. Usage: python3 validate.py <file1> <file2>")
    sys.exit(1)

# Loop through every line
with open(fileName1, 'r') as file1, open(fileName2, 'r') as file2:
    file1_lines = file1.readlines()
    file2_lines = file2.readlines()

# If all lines are the same, report that
if file1_lines == file2_lines:
    print("The files are identical.")
else:
    # Calculate the percentage difference
    percent_diff = calculate_difference_percentage(file1_lines, file2_lines)
    print(f"The files are different by {percent_diff:.2f}%")
    
    # Optional: Report which file is longer
    if len(file1_lines) > len(file2_lines):
        print(f"Note: File 1 has {len(file1_lines) - len(file2_lines)} more lines than File 2")
    elif len(file2_lines) > len(file1_lines):
        print(f"Note: File 2 has {len(file2_lines) - len(file1_lines)} more lines than File 1")
