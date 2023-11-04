import re
regex_sentence = "I love to eat pizza and pasta 3 times a day"


# Using a capture group to extract the words "pizza" and "pasta"
pattern = "((pizza)|(pasta))"
matches = re.findall(pattern, regex_sentence)


print("Captured words:", matches)
# Output: Captured words: [("pizza", "pizza", ""), ("pasta", "", "pasta")]

# The input text to match
text = "The quick brown fox jumps over the lazy dog 123 times."


# Write a regex pattern to match the following:
# All the digits in the text

# Solution: “\d”, [0-9]


text = "The quick brown fox jumps over the lazy dog 123 times."

# The re.search() function can be used to search for the first occurrence of a pattern in a given string. It returns a Match object if a match is found, or None if no match is found.
# Search for a digit in the text
match = re.search(r"\d", text)
if match:
    print("First digit found at position:", match.start())


# Output:
# First digit found at position: 42


# Split the text on punctuation marks
# The re.split() function can be used to split a string into a list of strings based on a specified pattern
result = re.split(r"[,;.]", text)
print("Words after splitting:", result)


# Output:
# Words after splitting: ['The quick', ' brown', ' fox jumps', ' over the lazy dog 123 times', '']


# Replace all digits with the word 'NUMBER'
# The re.sub() function can replace all occurrences of a pattern in a string with a specified replacement.
result = re.sub(r"\d+", "NUMBER", text)
print("Text after replacing digits:", result)


# Output:
# Text after replacing digits: The quick brown fox jumps over the lazy dog NUMBER times



# Find all the words in the text
# The re.findall() function can return a list of all the non-overlapping occurrences of a pattern in a string.
result = re.findall(r"\b\w+\b", text)
print("Words in the text:", result)


# Output:
# Words in the text: ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '123', 'times']



# Grouping Capturing groups allows us to extract specific parts of the matching pattern and refer to them later. Capturing groups are created by enclosing a portion of the pattern in parentheses.
x = re.search("I love to eat (\w+) and (\w+)", text)
print("group 1 is: " + x.group(1) + " group 2 is: " + x.group(2))


#Output: 
# group 1 is: pizza group 2 is: pasta



# Non-capturing groups are similar to capturing groups but don't capture the matching text. They are indicated by “(?:)”
x = re.search("I love to eat (?:pizza) and (\w+)", text)
print(x.group(1))
# Output: pasta


# Lookaheads allow you to match a pattern only if another pattern follows it. Positive lookaheads are indicated by (?=) and negative lookaheads are indicated by (?!).
x = re.search("\w+(?= times)", text)
print(x.group())
# output: 3 



# Backreferences allow you to reuse the text matched by a capturing group in the same pattern.
x = re.search("(\\w+) and \\1", text)
print(x.group())
# output: pizza and pizza


# Non-Capturing Groups
x = re.search("I love to eat (?:pizza) and (\w+)", text)


print(x.group(1))
# output: pasta
