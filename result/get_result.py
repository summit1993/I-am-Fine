import pickle

data_pkl = pickle.load(open('1.pkl', 'rb'))
test_pkl = pickle.load(open('test.pkl', 'rb'))

predicted = data_pkl['test_predictions']
ids = test_pkl['images']

f = open('Poker.csv', 'w')
f.write('id,predicted\n')

for i in range(len(ids)):
    id = ids[i]
    predict = predicted[i]
    predict = ' '.join([str(a) for a in predict])
    f.write(id + ',' + predict + '\n')

f.close()