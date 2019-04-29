import scipy.io as sio
import numpy as np


def read_deal_mat(file, save_file):
    with open(file) as f:
        lines = f.readlines()
    line_0 = lines[0]
    line_0 = line_0.split(' ')
    total_points, num_features, num_labels = int(line_0[0]), int(line_0[1]), int(line_0[2])
    feature_rows = []
    feature_cols = []
    feature_values = []
    label_rows = []
    label_cols = []
    label_values = []
    for i in range(1, len(lines)):
        line = lines[i]
        line = line.split()
        labels = line[0]
        labels = labels.split(',')
        # print(labels)
        for a in labels:
            label_rows.append(i)
            label_cols.append(int(a))
            label_values.append(1)
        for v in line[1:]:
            v = v.split(':')
            feature_rows.append(i)
            feature_cols.append(int(v[0]))
            feature_values.append(float(v[1]))
    sio.savemat(save_file,
                {'total_points':float(total_points),
                 'num_features': float(num_features),
                 'num_labels': float(num_labels),
                 'label_rows': np.array(label_rows, dtype=np.double),
                 'label_cols': np.array(label_cols, dtype=np.double),
                 'label_values': np.array(label_values, dtype=np.double),
                 'feature_rows': np.array(feature_rows, dtype=np.double),
                 'feature_cols': np.array(feature_cols, dtype=np.double),
                 'feature_values': np.array(feature_values)}
                )

if __name__ == '__main__':
    read_deal_mat('test.txt', 'test.mat')