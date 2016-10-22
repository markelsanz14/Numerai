import matplotlib.pyplot as plt

class visualizeData():

	def __init__(self, data):
		features = [entry[0] for entry in data]
		f1 = [entry[0] for entry in features]
		labels = [entry[1] for entry in data]
		plt.plot(f1,labels,'ro')
		plt.axis([-0.5,1.5,-0.5,1.5])
		plt.show()
