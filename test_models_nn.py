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
from sklearn import preprocessing
from keras.models import Sequential

def get_X_and_Y(input_path):
    num_samples = 0
    X = None
    Y = None
    for subdir, dirs, files in os.walk(input_path):
        for file_i in files:
            if("features" in file_i and "test" not in file_i):
                class_label = int(file_i[1])
                #print("class label is %d" % class_label)
                file_i_path = os.path.join(subdir, file_i)
                #print(file_i_path)
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
                    print("num lines")
                    print(num_lines)
                    Y = np.zeros((num_lines, 1))
                else:
                    print("num lines")
                    print(num_lines)
                    y_curr = class_label*np.ones((num_lines, 1))
                    print(y_curr)
                    Y = np.vstack((Y, y_curr))
                fd.close()
    print("Number of samples = %d" % num_samples)
    print("X shape")
    print("Y shape")
    print(X.shape)
    print(Y.shape)
    return X, Y

def get_X_and_Y_from_immediate_files_in_dir(input_path):
    num_samples = 0
    X = None
    Y = None
    for subdir, dirs, files in os.walk(input_path):
        for file_i in files:
            if("features" in file_i and "test" not in file_i):
                class_label = int(file_i[1])
                #print("class label is %d" % class_label)
                file_i_path = os.path.join(subdir, file_i)
                #print(file_i_path)
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
                    print("num lines")
                    print(num_lines)
                    Y = np.zeros((num_lines, 1))
                else:
                    print("num lines")
                    print(num_lines)
                    y_curr = class_label*np.ones((num_lines, 1))
                    print(y_curr)
                    Y = np.vstack((Y, y_curr))
                fd.close()
        break
    print("Number of samples = %d" % num_samples)
    print("X shape")
    print("Y shape")
    print(X.shape)
    print(Y.shape)
    return X, Y


def get_X_and_Y_and_filename_arr(input_path):
    num_samples = 0
    X = None
    Y = None
    filepath_arr = None
    for subdir, dirs, files in os.walk(input_path):
        for file_i in files:
            if("features" in file_i and "test" not in file_i):
                class_label = int(file_i[1])
                #print("class label is %d" % class_label)
                file_i_path = os.path.join(subdir, file_i)
                #print(file_i_path)
                fd = open(file_i_path, 'r')
                num_lines = 0
                for line in fd:
                    json_obj = json.loads(line)
                    features = np.asarray(json_obj["features"])
                    file_path_curr_file_arr = json_obj["name"].split('/')
                    filename_curr_file = np.asarray(file_path_curr_file_arr[-1], dtype=object)
                    if num_samples == 0:
                        X = features
                        filepath_arr = filename_curr_file
                    else:
                        X = np.vstack((X, features))
                        filepath_arr = np.vstack((filepath_arr, filename_curr_file))

                    num_samples += 1
                    num_lines += 1
                if class_label == 0:
                    print("num lines")
                    print(num_lines)
                    Y = np.zeros((num_lines, 1))
                else:
                    print("num lines")
                    print(num_lines)
                    y_curr = class_label*np.ones((num_lines, 1))
                    print(y_curr)
                    Y = np.vstack((Y, y_curr))
                fd.close()
    print("Number of samples = %d" % num_samples)
    print("X shape")
    print("Y shape")
    print(X.shape)
    print(Y.shape)
    return X, Y, filepath_arr

def get_X_and_Y_and_filename_arr_single_file(input_path):
    num_samples = 0
    X = None
    Y = None
    filepath_arr = None
    file_i = input_path.split('/')[-1]
    class_label = int(file_i[1])
    #print("class label is %d" % class_label)
    #file_i_path = os.path.join(subdir, file_i)
    #print(file_i_path)
    fd = open(input_path, 'r')
    num_lines = 0
    for line in fd:
        json_obj = json.loads(line)
        features = np.asarray(json_obj["features"])
        file_path_curr_file_arr = json_obj["name"].split('/')
        filename_curr_file = np.asarray(file_path_curr_file_arr[-1], dtype=object)
        if num_samples == 0:
            X = features
            filepath_arr = filename_curr_file
        else:
            X = np.vstack((X, features))
            filepath_arr = np.vstack((filepath_arr, filename_curr_file))

        num_samples += 1
        num_lines += 1
    if class_label == 0:
        print("num lines")
        print(num_lines)
        Y = np.zeros((num_lines, 1))
    else:
        print("num lines")
        print(num_lines)
        y_curr = class_label*np.ones((num_lines, 1))
        print(y_curr)
        Y = np.vstack((Y, y_curr))
    fd.close()
    print("Number of samples = %d" % num_samples)
    print("X shape")
    print("Y shape")
    print(X.shape)
    print(Y.shape)
    return X, Y, filepath_arr

def get_X_single_file(input_path):
    X = None
    fd = open(input_path, 'r')
    num_lines = 0
    for line in fd:
        json_obj = json.loads(line)
        features = np.asarray(json_obj["features"])
        if num_lines == 0:
            X = features
        else:
            X = np.vstack((X, features))
        num_lines += 1
    fd.close()
    print("X shape")
    print(X.shape)
    return X

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

def create_submission_file_probabilities(filename_vec, probabilities_matrix, test_submission_filename):
    with open(test_submission_filename, 'w') as outfd:
        csv_writer = csv.writer(outfd, delimiter=',')
        header = ['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
        csv_writer.writerow(header)
        # row_template = ['','0','0','0','0','0','0','0','0','0','0']
        print('WRITING PROBABILITIES TO FILE')
        count = 0
        print_step_size = 10000
        for i in range(len(probabilities_matrix)):
            if i%print_step_size == 0:
                print('Processing row number %d' % count)
                count += print_step_size
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
            #label = int(y_hat[i])
            # Decrease multi-class loss by strengthening labeling:
            # prob_vec[label] += 1.0
            # prob_vec = map(div_by_2, prob_vec)
            prob_vec = map(str, prob_vec)
            row_i_arr = []
            row_i_arr.append(filename_i)
            for i in prob_vec:
                row_i_arr.append(i)
            #prob_vec.insert(0, filename_i)
            #row_i_arr = prob_vec
            # print row_i_arr
            # row_i = ','.join(row_i_arr)
            csv_writer.writerow(row_i_arr)


def create_kaggle_submission_for_test_features(test_features_doc, test_submission_filename, model):
    fd = open(test_features_doc, 'r')
    num_samples = 0
    #batch_ind = 500
    for line in fd:
        json_obj = json.loads(line)
        features = np.asarray(json_obj["features"])
        print("processing file %s" % json_obj["name"])
        
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
    #X_test_filename = '/media/sf_ubuntu_vm/statefarm_data/feature_set_1st_2nd_hog_deriv/X_test_submission.npy'
    #filenames_vec_test = '/media/sf_ubuntu_vm/statefarm_data/feature_set_1st_2nd_hog_deriv/filenames_test_submission.npy'
    #np.save(X_test_filename, X_test)
    #np.save(filenames_vec_test, filename_vec)

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

# def create_kaggle_submission_for_test_features_nn(test_features_doc, test_submission_filename, model, scaler):
#     fd = open(test_features_doc, 'r')
#     num_samples = 0
#     batch_size = 500
#     for line in fd:
#
#         #if num_samples == 10:
#         #    break
#         if num_samples%batch_size == 0:
#             print('Loading sample number %d' % num_samples)
#
#         json_obj = json.loads(line)
#         features = np.asarray(json_obj["features"])
#         #print("processing file %s" % json_obj["name"])
#         if num_samples == 0:
#             X_test = features
#             filename_vec = np.empty([1,1], dtype=object)
#             filename_i = json_obj["name"].split('/')[-1]
#             filename_vec[0] = filename_i
#         else:
#             filename_vec_i = np.empty([1,1], dtype=object)
#             filename_i = json_obj["name"].split('/')[-1]
#             filename_vec_i[0] = filename_i
#             X_test = np.vstack((X_test, features))
#             filename_vec = np.vstack((filename_vec, filename_vec_i))
#         num_samples += 1
#     fd.close()
#     #X_test_filename = '/media/sf_ubuntu_vm/statefarm_data/feature_set_1st_2nd_hog_deriv/X_test_submission.npy'
#     #filenames_vec_test = '/media/sf_ubuntu_vm/statefarm_data/feature_set_1st_2nd_hog_deriv/filenames_test_submission.npy'
#     #np.save(X_test_filename, X_test)
#     #np.save(filenames_vec_test, filename_vec)
#     # X_test_filename = '/media/sf_ubuntu_vm/statefarm_data/feature_set_1st_2nd_hog_deriv/X_test_submission.npy'
#     # filenames_vec_test = '/media/sf_ubuntu_vm/statefarm_data/feature_set_1st_2nd_hog_deriv/filenames_test_submission.npy'
#     # X_test = np.load(X_test_filename)
#     # filename_vec = np.load(filenames_vec_test)
#     # print 'dim check:'
#     # print X_test.shape
#     # print filename_vec.shape
#     print('BEFORE STANDARDIZATION')
#     X_test_scaled = scaler.transform(X_test)
#     print('AFTER STANDARDIZATION')
#     #y_hat = model.predict(X_test_scaled)
#     probabilities_matrix = model.predict_proba(X_test_scaled)
#     # create_submission_file(filename_vec, y_hat, test_submission_filename)
#     create_submission_file_probabilities(filename_vec, probabilities_matrix, test_submission_filename)


def create_kaggle_submission_for_test_features_nn(test_features_doc, test_submission_filename, model, scaler):
    fd = open(test_features_doc, 'r')
    mini_batch_size = 5000
    batch_size = 10000
    num_lines = get_num_lines(fd)
    fd.seek(0)
    lines_arr = fd.readlines()
    fd.close()
    prob_matrix_full = None
    filename_vec = None
    prob_mat_size_count = 0
    curr_line = prob_mat_size_count * batch_size
    while (curr_line < num_lines):
        print('Processing batch starting with line number %d' % curr_line)
        X_test_scaled_i, filename_vec_i = get_X_test_scaled(prob_mat_size_count, batch_size, mini_batch_size, lines_arr,
                                                            num_lines, scaler)
        # y_hat = model.predict(X_test_scaled_i)
        # probabilities_matrix = None
        probabilities_matrix = model.predict_proba(X_test_scaled_i)
        for i in range(probabilities_matrix.shape[0]):
            total = sum(probabilities_matrix[i, :])
            # print(total)
            # print(probabilities_matrix[i,:])
            probabilities_matrix[i, :] /= total
            # print(probabilities_matrix[i,:])
        if prob_mat_size_count == 0:
            prob_matrix_full = probabilities_matrix
            filename_vec = filename_vec_i
        else:
            prob_matrix_full = np.vstack((prob_matrix_full, probabilities_matrix))
            filename_vec = np.vstack((filename_vec, filename_vec_i))
        prob_mat_size_count += 1
        curr_line = prob_mat_size_count * batch_size
    # probabilities_matrix = model.predict_proba(X_test_scaled)
    # create_submission_file(filename_vec, y_hat, test_submission_filename)
    create_submission_file_probabilities(filename_vec, prob_matrix_full, test_submission_filename)

def get_num_lines(fd):
    count = 0
    for line in fd:
        count += 1
    return count

def get_X_test_scaled(data_ind, batch_size, mini_batch_size, lines_arr, num_lines, scaler):
    offset = data_ind*batch_size
    filename_vec = np.empty([1, 1], dtype=object)
    filename_vec_i = np.empty([1, 1], dtype=object)
    for batch_inx in range(batch_size):
        inx = offset + batch_inx
        if(inx == num_lines):
            break
        line = lines_arr[inx]
        if batch_inx%mini_batch_size == 0:
            print('Loading sample number %d' % inx)
        json_obj = json.loads(line)
        features = np.asarray(json_obj["features"])
        filename_i = json_obj["name"].split('/')[-1]
        #print("processing file %s" % json_obj["name"])
        if batch_inx == 0:
            X_test = features
            filename_vec[0] = filename_i
        else:
            filename_vec_i[0] = filename_i
            X_test = np.vstack((X_test, features))
            filename_vec = np.vstack((filename_vec, filename_vec_i))
        # num_samples += 1
    X_test_scaled = scaler.transform(X_test)
    return X_test_scaled, filename_vec
    
def create_kaggle_submission_for_test_features_nn_ensemble(test_features_doc, test_submission_filename, model_arr, scaler):
    fd = open(test_features_doc, 'r')
    mini_batch_size = 5000
    batch_size = 10000
    num_lines = get_num_lines(fd)
    fd.seek(0)
    lines_arr = fd.readlines()
    fd.close()
    prob_matrix_full = None
    filename_vec = None
    prob_mat_size_count = 0
    curr_line = prob_mat_size_count*batch_size
    while(curr_line < num_lines):
        print('Processing batch starting with line number %d' % curr_line)
        X_test_scaled_i, filename_vec_i = get_X_test_scaled(prob_mat_size_count, batch_size, mini_batch_size, lines_arr, num_lines, scaler)
        #y_hat = model.predict(X_test_scaled_i)
        probabilities_matrix = None
        count = 0
        for model_i in model_arr:
            prob_i = model_i.predict_proba(X_test_scaled_i)
            if count == 0:
                probabilities_matrix = prob_i
            else:
                probabilities_matrix += prob_i
            count += 1
        for i in range(probabilities_matrix.shape[0]):
            total = sum(probabilities_matrix[i,:])
            #print(total)
            #print(probabilities_matrix[i,:])
            probabilities_matrix[i,:] /= total
            #print(probabilities_matrix[i,:])
        if prob_mat_size_count == 0:
            prob_matrix_full = probabilities_matrix
            filename_vec = filename_vec_i
        else:
            prob_matrix_full = np.vstack((prob_matrix_full, probabilities_matrix))
            filename_vec = np.vstack((filename_vec, filename_vec_i))
        prob_mat_size_count += 1
        curr_line = prob_mat_size_count*batch_size
    #probabilities_matrix = model.predict_proba(X_test_scaled)
    # create_submission_file(filename_vec, y_hat, test_submission_filename)
    create_submission_file_probabilities(filename_vec, prob_matrix_full, test_submission_filename)
    
def create_kaggle_submission_for_test_features_nn_ensemble_supp(test_features_doc, test_submission_filename,
                                                                model_arr, model_2, scaler, model_2_weight):
    fd = open(test_features_doc, 'r')
    mini_batch_size = 5000
    batch_size = 10000
    num_lines = get_num_lines(fd)
    fd.seek(0)
    lines_arr = fd.readlines()
    fd.close()
    prob_matrix_full = None
    filename_vec = None
    prob_mat_size_count = 0
    curr_line = prob_mat_size_count*batch_size
    scaling_factor_model_2 = model_2_weight
    print('Scaling factor for model 2 = %d' % scaling_factor_model_2)
    while(curr_line < num_lines):
        print('Processing batch starting with line number %d' % curr_line)
        X_test_scaled_i, filename_vec_i = get_X_test_scaled(prob_mat_size_count, batch_size, mini_batch_size, lines_arr, num_lines, scaler)
        #y_hat = model.predict(X_test_scaled_i)
        probabilities_matrix = None
        count = 0
        for model_i in model_arr:
            prob_i = model_i.predict_proba(X_test_scaled_i)
            if count == 0:
                probabilities_matrix = prob_i
            else:
                probabilities_matrix += prob_i
            count += 1
        probabilities_matrix += scaling_factor_model_2*(model_2.predict_proba(X_test_scaled_i))
        for i in range(probabilities_matrix.shape[0]):
            total = sum(probabilities_matrix[i,:])
            #print(total)
            #print(probabilities_matrix[i,:])
            probabilities_matrix[i,:] /= total
            #print(probabilities_matrix[i,:])
        if prob_mat_size_count == 0:
            prob_matrix_full = probabilities_matrix
            filename_vec = filename_vec_i
        else:
            prob_matrix_full = np.vstack((prob_matrix_full, probabilities_matrix))
            filename_vec = np.vstack((filename_vec, filename_vec_i))
        prob_mat_size_count += 1
        curr_line = prob_mat_size_count*batch_size
    #probabilities_matrix = model.predict_proba(X_test_scaled)
    # create_submission_file(filename_vec, y_hat, test_submission_filename)
    create_submission_file_probabilities(filename_vec, prob_matrix_full, test_submission_filename)
    


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

    print("TRAIN SET SHAPE:")
    print(X_train.shape)
    print(y_train.shape)
    print(X_train)
    print(y_train)

    print("TEST SET SHAPE:")
    print(X_test.shape)
    print(y_test.shape)

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

    print(model.predict_proba(X_test))
    print(y_test)
    print(y_hat)

    print("ACCURACY = %f" % accuracy)
    print("PRECISION = %f" % precision)
    print("RECALL = %f" % recall)
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

    print("------------------ %f minutes elapsed ------------------------" % ((time.time() - start_time)/60.0))



