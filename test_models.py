import numpy as np
from sklearn import cross_validation
import os
import json
import time
from sklearn import ensemble
from sklearn import metrics
from sklearn import tree
import csv
from sklearn import naive_bayes
from sklearn import svm
from sklearn import linear_model

def get_X_and_Y(input_path):
    num_samples = 0
    for subdir, dirs, files in os.walk(input_path):
        for file_i in files:
            if("features" in file_i and "test" not in file_i):
                class_label = int(file_i[1])
                print "class label is %d" % class_label
                file_i_path = os.path.join(subdir, file_i)
                print file_i_path
                fd = open(file_i_path, 'r')
                num_lines = 0
                for line in fd:
                    json_obj = json.loads(line)
                    features = np.asarray(json_obj["features"])
                    if num_samples == 0:
                        X = features
                    else:
                        X = np.vstack((X, features))
                    num_samples += 1
                    num_lines += 1
                if class_label == 0:
                    print "num lines"
                    print num_lines
                    Y = np.zeros((num_lines, 1))
                else:
                    print "num lines"
                    print num_lines
                    y_curr = class_label*np.ones((num_lines, 1))
                    print y_curr
                    Y = np.vstack((Y, y_curr))
                fd.close()
    print "Number of samples = %d" % num_samples
    print "X shape"
    print "Y shape"
    print X.shape
    print Y.shape
    return X, Y

def create_submission_file(filename_vec, y_hat, test_submission_filename):
    with open(test_submission_filename, 'w') as outfd:
        csv_writer = csv.writer(outfd, delimiter=',')
        header = ['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
        csv_writer.writerow(header)
        # row_template = ['','0','0','0','0','0','0','0','0','0','0']
        for i in range(len(y_hat)):
            # print "filename:"
            filename_i = str(filename_vec[i][0])
            # print filename_i
            # print "label:"
            label = int(y_hat[i])
            # print label
            row_i_arr = ['','0','0','0','0','0','0','0','0','0','0']
            row_i_arr[label + 1] = "1.0"
            row_i_arr[0] = filename_i
            # print row_i_arr
            # row_i = ','.join(row_i_arr)
            csv_writer.writerow(row_i_arr)

def div_by_2(x):
    return float(x)/2.0

def create_submission_file_probabilities(filename_vec, probabilities_matrix, y_hat, test_submission_filename):
    with open(test_submission_filename, 'w') as outfd:
        csv_writer = csv.writer(outfd, delimiter=',')
        header = ['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
        csv_writer.writerow(header)
        # row_template = ['','0','0','0','0','0','0','0','0','0','0']
        for i in range(len(probabilities_matrix)):
            # print "filename:"
            filename_i = str(filename_vec[i][0])
            # print filename_i
            # print "label:"
            # print label
            # row_i_arr = ['','0','0','0','0','0','0','0','0','0','0']
            # row_i_arr[label + 1] = "1.0"
            # row_i_arr[0] = filename_i
            prob_vec = probabilities_matrix[i, :]
            prob_vec = prob_vec.tolist()
            label = int(y_hat[i])
            # Decrease multi-class loss by strengthening labeling:
            # prob_vec[label] += 1.0
            # prob_vec = map(div_by_2, prob_vec)
            prob_vec = map(str, prob_vec)
            prob_vec.insert(0, filename_i)
            row_i_arr = prob_vec
            # print row_i_arr
            # row_i = ','.join(row_i_arr)
            csv_writer.writerow(row_i_arr)


def create_kaggle_submission_for_test_features(test_features_doc, test_submission_filename, model):
    fd = open(test_features_doc, 'r')
    num_samples = 0
    for line in fd:
        json_obj = json.loads(line)
        features = np.asarray(json_obj["features"])
        print "processing file %s" % json_obj["name"]
        if num_samples == 0:
            X_test = features
            filename_vec = np.empty([1,1], dtype=object)
            filename_i = json_obj["name"].split('/')[-1]
            filename_vec[0] = filename_i
        else:
            filename_vec_i = np.empty([1,1], dtype=object)
            filename_i = json_obj["name"].split('/')[-1]
            filename_vec_i[0] = filename_i
            X_test = np.vstack((X_test, features))
            filename_vec = np.vstack((filename_vec, filename_vec_i))
        num_samples += 1
    fd.close()
    X_test_filename = '/media/sf_ubuntu_vm/statefarm_data/feature_set_1st_2nd_hog_deriv/X_test_submission.npy'
    filenames_vec_test = '/media/sf_ubuntu_vm/statefarm_data/feature_set_1st_2nd_hog_deriv/filenames_test_submission.npy'
    np.save(X_test_filename, X_test)
    np.save(filenames_vec_test, filename_vec)

    # X_test_filename = '/media/sf_ubuntu_vm/statefarm_data/feature_set_1st_2nd_hog_deriv/X_test_submission.npy'
    # filenames_vec_test = '/media/sf_ubuntu_vm/statefarm_data/feature_set_1st_2nd_hog_deriv/filenames_test_submission.npy'
    # X_test = np.load(X_test_filename)
    # filename_vec = np.load(filenames_vec_test)
    # print 'dim check:'
    # print X_test.shape
    # print filename_vec.shape
    y_hat = model.predict(X_test)

    probabilities_matrix = model.predict_proba(X_test)
    # create_submission_file(filename_vec, y_hat, test_submission_filename)
    create_submission_file_probabilities(filename_vec, probabilities_matrix, y_hat, test_submission_filename)


if __name__ == "__main__":
    start_time = time.time()
    # input_path = "/media/sf_ubuntu_vm/statefarm_data/feature_set_100_hog_bins"
    input_path = "/media/sf_ubuntu_vm/statefarm_data/feature_set_1st_2nd_hog_deriv"

    # X_filename = 'statefarm_X.npy'
    # Y_filename = 'statefarm_Y.npy'

    X_filename = 'statefarm_X_1st_2nd_hog.npy'
    Y_filename = 'statefarm_Y_1st_2nd_hog.npy'

    # GET X and Y from json object files:
    X, Y = get_X_and_Y(input_path)
    np.save(X_filename, X)
    np.save(Y_filename, Y)

    # GET X and Y from saved numpy arrays:
    # X = np.load(X_filename)
    # Y = np.load(Y_filename)

    # SPLIT into train and test:
    test_frac = 0.3
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=test_frac, random_state=0)

    print "TRAIN SET SHAPE:"
    print X_train.shape
    print y_train.shape
    print X_train
    print y_train

    print "TEST SET SHAPE:"
    print X_test.shape
    print y_test.shape

    # TRAIN AND CROSS VALIDATE MODELS:
    model = ensemble.RandomForestClassifier(n_estimators=255)
    # model = naive_bayes.MultinomialNB()
    # model = svm.SVC()
    max_tree_depth = 30
    # model = ensemble.AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=max_tree_depth), n_estimators=600, algorithm="SAMME.R", random_state=0)
    # model.fit(X_train, y_train)
    # Use full train data for submission:
    model.fit(X, Y)

    y_hat = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_hat)
    precision = metrics.precision_score(y_test, y_hat)
    recall = metrics.recall_score(y_test, y_hat)

    print model.predict_proba(X_test)
    print y_test
    print y_hat

    print "ACCURACY = %f" % accuracy
    print "PRECISION = %f" % precision
    print "RECALL = %f" % recall
    # Use full train data for submission:
    # model = ensemble.RandomForestClassifier(n_estimators=255)
    # model = ensemble.AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=max_tree_depth), n_estimators=255, algorithm="SAMME.R", random_state=0)
    # model.fit(X, Y)

    # print "CROSS VALIDATION:"
    # num_folds = 5
    # cross_val_arr = cross_validation.cross_val_score(model, X, Y.flatten(), cv=num_folds)
    # mean_cross_val_score = sum(cross_val_arr)/float(len(cross_val_arr))
    # print "MEAN %d-FOLD CROSS VALIDATION SCORE = %f" % (num_folds, mean_cross_val_score)

    # OUTPUT TEST RESULTS FOR KAGGLE:
    test_features_doc = '/media/sf_ubuntu_vm/statefarm_data/feature_set_1st_2nd_hog_deriv/test_features'
    test_submission_filename = '/media/sf_ubuntu_vm/statefarm_data/feature_set_1st_2nd_hog_deriv/test_submission.csv'
    create_kaggle_submission_for_test_features(test_features_doc, test_submission_filename, model)

    print "------------------ %f minutes elapsed ------------------------" % ((time.time() - start_time)/60.0)



