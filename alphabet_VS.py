low_char = "abcdefghijklmnopqrstuvwxyz"
up_char = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

sentence = input("enter a sentence: ")
shift = int(input("enter a number you want to shift: "))

result = ""

for char in sentence:
    if char in low_char:
        index = low_char.index(char)
        new_char = low_char[(index + shift) - 26]
        result += new_char
    elif char in up_char:
        index = up_char.index(char)
        new_char = up_char[(index + shift) - 26]
        result += new_char
    else:
        result += char
        
print(result)


