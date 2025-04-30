import pickle


def load_and_print_matrices(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        
    print(data)
    

if __name__ == "__main__":
    file_path = 'state-space-serialized.pkl'
    load_and_print_matrices(file_path)