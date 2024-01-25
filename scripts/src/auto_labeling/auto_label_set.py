from segmentation import *
import warnings
import sys
import pickle



def main(image_folder, clf_path):
    with open(clf_path, 'rb') as handle:
        clf = pickle.load(handle)
    predict_apex(image_folder, clf)
    
    
warnings.filterwarnings("ignore", category=UserWarning)
image_folder, clf_path = sys.argv[1:3]

if __name__ == "__main__":
    main(image_folder, clf_path)