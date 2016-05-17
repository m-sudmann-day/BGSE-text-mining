import re

def MyRegex(pattern, text):

    matches = re.findall(pattern, text, re.MULTILINE|re.DOTALL)
    
    if len(matches) == 1:
        return matches[0]
    else:
        return None
