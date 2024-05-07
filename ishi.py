# pickling the model
import pickle

import classifier as classifier

pickle_out = open("classifier.pkl", "wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()
