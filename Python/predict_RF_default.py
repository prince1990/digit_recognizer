import csv as csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier

train_file = csv.reader( open("/home/prince/Kaggle/Digit Recognision/train.csv"))
train_file.next()  #skip headers

label = []
train = []

for line in train_file:
	label.append(line[0])
	train.append(line[1:])
#	print line

#print train

train = np.array(train)
label = np.array(label)

rf = RandomForestClassifier()

rf.fit(train,label)

test_file = csv.reader( open("/home/prince/Kaggle/Digit Recognision/test.csv"))
test_file.next()

test = []

for line in test_file:
	test.append(line)

test = np.array(test)
output = rf.predict(test)
ids = np.array( range(1,28001))

output_file = csv.writer( open("output.csv", "wb"))
output_file.writerow(["ImageId", "Label"])
output_file.writerows(zip(ids, output))
