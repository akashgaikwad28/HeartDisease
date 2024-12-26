import pickle

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def make_predictions(model, data):
    return model.predict(data)
