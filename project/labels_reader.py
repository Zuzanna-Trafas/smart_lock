import pickle

with open('labels', 'rb') as f: # load dictionary with labels
	dict = pickle.load(f)
	f.close()

for name, value in dict.items():
    print(name)
    print(value)
    print("\n")