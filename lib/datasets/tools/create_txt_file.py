import os

with open('train.txt', 'w') as f:
    for file in os.listdir():
        if file == 'create_txt_file.py':
            continue
        else:
            f.write(file + "\n")
