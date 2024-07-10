# src/utils.py
def load_data(reviews_path, labels_path):
    with open(reviews_path, 'r', encoding='utf-8') as file:
        reviews = file.read().splitlines()

    with open(labels_path, 'r', encoding='utf-8') as file:
        labels = file.read().splitlines()

    return reviews, labels
