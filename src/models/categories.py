from sklearn.mixture import GaussianMixture
import pickle


def train_model(data: [[int]], labels = 1):
    #create a new model and train it

    model = GaussianMixture(n_components = labels, max_iter = 500)

    model.fit(data)

    pickle.dump(model, 'trained_models/gaussianMixture.txt')    
    
def load_model():
    return pickle.load('trained_models/gaussianMixture.txt')

def label_article(data):
    model = load_model()

    return model.predict(data)
    
    
