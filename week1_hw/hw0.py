###1

# 1 use open a connection to the file
# 2 read contents of the file
# 3 use the json module to read the string as a list
#(replace None with relevant code)

import json
import sys

# Assert that we are running version 2.7 of Python.
if (sys.version_info.major != 2 or sys.version_info.minor != 7):
    message = "Python version 2.7 required.  This is "
    message += str(sys.version_info.major) + "."
    message += str(sys.version_info.minor) + "."
    raise RuntimeError(message)

# Read the file content.
file_handle = open("1977_Ford_Matthew.txt")
file_content = file_handle.read()
file_handle.close()

# Extract the content as JSON and get a copy of the speech text.
speech = json.loads(file_content)[2]

print "The speech text contains {0} characters.".format(len(speech))

###2

#1 

stripped_text = speech

# For each nonalphanumeric character, replace with a space.  This is
# safer than replacing with an empty string because some punctuation
# separates words without a space, i.e. '--'.
for char in ',.:;[]"?$:-':
    stripped_text = stripped_text.replace(char, ' ')

#2

# Split the string into words.
word_list = stripped_text.split(' ')

# Because of the way the punctuation was replaced with spaces, there are
# instances of multiple adjacent spaces.  Therefore, empty strings appear
# in the word list.  Remove these.
word_list = [word for word in word_list if word != '']

#3 

# Convert all words to lower case.
lower_words = [word.lower() for word in word_list]

print "The speech contains {0} words.".format(len(lower_words))

###3

def preprocess(text, nonalphanum_chars, lower = True):

    # For each nonalphanumeric character, replace with a space.  This
    # is safer than replacing with an empty string because some punctuation
    # separates words without a space, i.e. '--'.
    for char in nonalphanum_chars:
        text = text.replace(char, ' ')

    # Split the string into words.
    words = text.split(' ')

    # Because of the way the punctuation was replaced with spaces, there
    # are instances of multiple adjacent spaces.  Therefore, empty strings
    # appear in the word list.  Remove these.
    words = [word for word in words if word != '']
    
    if lower:
        # Convert all words to lower case.
        words = [word.lower() for word in words]

    return words

# Preprocess (strip, tokenize, and lower) the speech into a list of words. 
words = preprocess(speech, ',.:;[]"?$:-', True)

print(words)

###4

from collections import OrderedDict

# Define a function for getting a list of unique word counts within
# text.  Return this list ordered from most frequent to least frequent.
def wordfreq(text):

    # Preprocess, removing the special characters passed in.
    words = preprocess(text, ',.:;[]"?$:-', True)

    # Using 'set', get a list of unique words.  Then count them into
    # a new dictionary.
    counts = dict([(word, words.count(word)) for word in set(words)])

    # Sort the list in reverse order. (t[0] is the key; t[1] is the value)
    sorted_counts = sorted(counts.items(), key=lambda t: t[1], reverse=True)

    # Preserve this sort order in an ordered dictionary.
    ordered_counts = OrderedDict(sorted_counts)

    return ordered_counts

word_frequencies = wordfreq(speech)

text = "The fourth most frequent word in the speech was '{0}'."
print text.format(word_frequencies.items()[3][0])

###5

# Define a function to get the frequencies of all presented word lengths.
def get_word_length_frequencies(word_list):

    # Initialize our dictionary of word frequencies.
    word_length_frequencies = {}

    # For each word in the original list...
    for word in word_list:

        # ...calculate its length.
        l = len(word)

        # If the length has already been encountered increment it,
        # otherwise create a new entry with a count of 1.
        if l in word_length_frequencies.keys():
            word_length_frequencies[l] = word_length_frequencies[l] + 1
        else:
            word_length_frequencies[l] = 1

    return word_length_frequencies

print "The word length frequences are:"
print get_word_length_frequencies(words)

###6

import re

# Defining a function for separating the speeches in presented text
# using the provided regular expression which is expected to define
# named groups for 'year' and 'president'.
def separate_speeches(text, pattern):

    # Initialize local vars.
    speeches = []
    year = -1

    # Precompile the regular expression for performance.
    pattern = re.compile(pattern)

    # Loop through each line in the text...
    for line in text:

        # ...see if it matches the regular expression that identifies the
        # header of a new speech.
        m = pattern.match(line)

        # If it matches..
        if (m != None):
            # ...and if we have a valid year, append all speech content accrued
            # thus far.
            if (year > 0):
                speeches.append([year, president, current_speech]) 
            # Store the year, president, and reset the speech content.
            year = m.group('year')
            president = m.group('president').replace('_', ' ')
            current_speech = []
        # If it doesn't match...
        else:
            # Strip the line and if there's anything left, append it to the
            # current speech.
            line = line.strip()
            if (len(line) > 0):
                current_speech.append(line)

    return speeches

# Load all State of the Union speeches.
file_handle = open("../data/pres_speech/sou_all.txt")
file_content = file_handle.readlines()
file_handle.close()

# Define the regular expression for the lines that indicate the start
# of a speech.
pattern = "\*+_(?P<year>[0-9]{4})_[^_]*_(?P<president>.*)_\*"

# Separate the speeches using the function defined above.
speeches = separate_speeches(file_content, pattern)

# Print metadata for each speech to output.
for s in speeches:
    print "{0}, {1}, {2} paragraphs".format(s[0], s[1], len(s[2]))

# Write the speeches to file in JSON format.
with open('output.json', 'w') as f:
     json.dump(speeches, f)

print
print "All speeches have been written to 'output.json'."
