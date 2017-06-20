import glob
import unicodedata
import string

# Constants
allowed_chars = string.ascii_letters + " .,;'"
n_chars = len(allowed_chars)

# Converts unicode strings into strings containing only ascii
# Characters. Accented letters will have their accents
# removed and other unicode characters will be removed
def unicode_to_ascii(s):
    return "".join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != "Mn"
            and c in allowed_chars
        )

# Returns a list of paths matching a given pattern
def find_files(path):
    return glob.glob(path)

# Opens a file of the given name and
# returns a list of the lines in ascii format
def read_lines(filename):
    with open(filename, encoding="UTF-8") as lines:
        lines = lines.read().strip().split('/n')
        return [unicode_to_ascii(line) for line in lines]

# Declare data containers
category_lines = {}
categories = []

# Load data into containers
for filename in find_files("name_data/names/*.txt"):
    category = filename.split('/')[-1].split('.')[0]
    categories.append(category)
    lines = read_lines(filename)
    category_lines[category] = lines

n_categories = len(categories)
