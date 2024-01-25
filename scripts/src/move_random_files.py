import random
random.seed(42)
import os
import sys

def main(input_dir, output_dir, p):
    try:
        # Create output_dir if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Copy p % of files from input_dir to output_dir
        # Get list of files in input_dir
        files = os.listdir(input_dir)
        # Get number of files to move
        n = int(len(files) * p)
        # Get random files
        random_files = random.sample(files, n)
        # Move files
        for file in random_files:
            os.rename(os.path.join(input_dir, file), os.path.join(output_dir, file))
    
    except Exception as e:
        print(e)
        sys.exit(1)
    else:
        print("Files moved successfully")




if __name__ == '__main__':
    # Get input and output directories from sys arg
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    # Get percentage of files to move 
    p = float(sys.argv[3])

    main(input_dir, output_dir, p)

