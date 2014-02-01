#!usr/bin/python3
import numpy as np
import pandas as pd
from collections import OrderedDict
from collections import Counter
import math
import copy

global complete_dict
complete_dict = {}
global class_count
class_count = {}
global class_prob
class_prob = {}
global class_weight
class_weight = {}

def predict_LR(path,train_matrix,count_variables,classy_list):
    #print('inside predict_lr')
    #print('calling cost function')
    class_weight_matrix = cost_function(train_matrix,classy_list)
    #print('the class_weight in test fn is',class_weight_matrix)
    test = pd.read_csv(path,sep='\t',names=['num','class','desc'],header=None)
    total_files = test['num'].count()
    #test_matrix = np.zeros((total_files,len(count_variables)))
    #class_list = list(test.ix[:,'class'].unique())
    #test_matrixy = numpy_matrix(test,count_variables,class_list,test_matrix)
    #print('the count of test file is',total_files)
    test_matrix = numpy_matrix(test,count_variables,total_files)
    '''Add X0 column in train_matrix
    '''
    x0_vector = np.ones((np.shape(test_matrix)[0],1))
    test_matrix = np.column_stack((x0_vector,test_matrix))
    #print('the class weight_matrix is ',class_weight_matrix)
    #print(np.shape(class_weight_matrix),np.shape(test_matrix))
    test_file = open('test_file.txt','w')
    for i in range(len(test_matrix)):
        for j in range(np.shape(test_matrix)[1]):
            test_file.write(str(test_matrix[i][j]))
            test_file.write(",")
        test_file.write("\n")
    test_output = np.dot(class_weight_matrix,np.transpose(test_matrix))
    #print(test_output,np.shape(test_output))
    max_value_index = np.argmax(test_output, 0)
    #print(max_value_index)
    #print(test['class'])
    test['class'] = test['class'].replace('student',0)
    test['class'] = test['class'].replace('faculty',1)
    test['class'] = test['class'].replace('project',2)
    test['class'] = test['class'].replace('course',3)
    test_y = np.array(test['class'])
    #print('test_y',test_y)
    test_list = []
    [test_list.append(class_weight[max_value_index[i]]) for i in range(len(max_value_index))]
    #print('class_weight is',class_weight)
    print(Counter(test_list))
    #print('the number of correct class is',np.count_nonzero(max_value_index == test_y),'the len of test_matrix is', len(test_matrix),'the accuracy is',np.count_nonzero(max_value_index == test_y) / len(test_matrix))
    print('mean is',np.mean(max_value_index == test_y)*100)
def cost_function(np_matrix,classy_list):
    #print('in cost function')
    alpha = 0.01
    lambda_value = 1
    #print('the np matrx in cost function is',np.shape(np_matrix))
    #class_weight = {}
    
    class_list = classy_list.unique().tolist()
    #print('the len of class_list is',len(class_list),np.shape(np_matrix)[1])
    class_weight_matrix = np.zeros((len(class_list),np.shape(np_matrix)[1]))
    #print('the class_weight_matrix shape is',np.shape(class_weight_matrix))
    count = 0
    for each_class in class_list:
        print(each_class)
        classy_matrix = np.array(classy_list.apply(lambda x: 1 if x==each_class else 0))[np.newaxis]
        #print('the shape of classy_matrix is',classy_matrix,np.shape(classy_matrix))
        theta_vector = np.random.rand(1,np.shape(np_matrix)[1])
        #old_theta_vector = np.random.rand(1,np.shape(np_matrix)[1])
        #print('the shape of theta vector is',np.shape(theta_vector))
        cc= 1
        for i in range(500):
            #print(cc)
            cc += 1
            hypothesis = np.dot(theta_vector,np.transpose(np_matrix))
            sigmoid = 1 / (1 + np.exp(-1 * hypothesis))
            '''
            #print('the shape of sigmoid is',np.shape(sigmoid))
            loss = (alpha * np.dot((sigmoid - classy_matrix),np_matrix)) / len(np_matrix)
            theta_vector = (1 + ((alpha * lambda_value)/len(np_matrix))) * theta_vector + loss
            #print('the theta_Vector is ',np.shape(theta_vector))
            '''
            #print('classy_matrix',classy_matrix)
            theta_vector = theta_vector - alpha * np.add((np.dot(np.subtract(sigmoid,classy_matrix),np_matrix)/len(np_matrix)),((lambda_value/len(np_matrix)) * theta_vector))
        #print('the theta vector is',each_class,theta_vector)
        class_weight[count] = copy.deepcopy(each_class)
        class_weight_matrix[count] = copy.deepcopy(theta_vector)
        count = count + 1 
    #print('the class_weight is',class_weight)
    return class_weight_matrix
        
  
def numpy_matrix(panda_df,count_variables,row_count):
    #print('inside np_matrix')
    desc_list = panda_df['desc'].tolist()
    
    np_matrix = np.zeros((row_count,len(count_variables)))
    #print('the np_matrix is',np.shape(np_matrix))
    count_variables_list = list(count_variables)
    line_number = 0
    for line in desc_list:
        #print(line_number)
        if type(line) is not str:
            line_number += 1
            continue
        elem = line.split()
        #print(elem)
        line_counter = Counter(elem)
        #print(line_counter)
        for word in elem:
            #print(word)
            '''
            Get the index of the word in count_variables
            '''
            if word in count_variables_list:
                word_index = count_variables_list.index(word)
                #print('the word index is',word_index,'the line number is',line_number,'the word is',word,'the line counter word is ',line_counter[word])
                np_matrix[line_number][word_index] = line_counter[word]
                #print('np_matrix[',line_number,'][',word_index,']', np_matrix[line_number][word_index])
        line_number += 1
    #print('the np_matrix is',np_matrix)

    return np_matrix
    
       
def train_read_files(path):
    '''
    The Training File is parsed by Pandas
    with header - none 
    and column - num, class,desc
    '''
    training = pd.read_csv(path,sep='\t',names=['num', 'class', 'desc'], header=None)
    '''
    To avoid the manuall indexing of 0,1,2 in pandas 
    a column is set a index but 
    the column is also avail to filter by setting drop-false
    '''
    training = training.set_index('class', drop = False)
    '''
    The unique values in 'class' column is filtered
    and saved in class_list - type is list
    '''
    class_list = list(training.ix[:,'class'].unique())
    complete_desc_list = []
    for classes in class_list:
        '''
        The desc column is stripped according 
        to each class values
        '''
        class_table = training[training['class'] == classes]
        class_count[classes] = class_table['class'].count()
        
        class_desc = class_table['desc'].tolist()
        '''
        Now the desc of each class is flatten 
        using list.extend,str.strip,str.split functions
        '''
        desc_list = []
        for elem in class_desc:
            if type(elem) is str:
                desc_list.extend(elem.strip().split(' '))
        complete_desc_list.extend(desc_list)
        count_variables = Counter(desc_list)
        complete_dict[classes] = count_variables
        
        
    count_variables = Counter(complete_desc_list)
    #print('the count_variables',count_variables)
    #print('the length of count_variables',len(count_variables))
    #print('the class count is',class_count)
    #print('the len of class_count is',len(class_count))
    
    '''NaiveBayes(class_list,count_variables,'TestSet\Final_project_test.txt')'''
    #train_np = np.zeros((sum(class_count.values()),len(count_variables)))
    #print('the shape of train_np is',np.shape(train_np))
    #train_matrix = np.zeros((sum(class_count.values()),len(count_variables)))
    train_matrix = numpy_matrix(training,count_variables,sum(class_count.values()))
    #print('the train_npy is',train_matrix)
    #print(np.count_nonzero(train_matrix))
    '''
    Add X0 column in train_matrix
    '''
    x0_vector = np.ones((sum(class_count.values()),1))
    train_matrix = np.column_stack((x0_vector,train_matrix))
    #print('the shape of train_matrix is ',np.shape(train_matrix))
    #print('the train_matrix is',train_matrix)
    infile = open('train_file.txt','w')
    for i in range(len(train_matrix)):
        for j in range(np.shape(train_matrix)[1]):
            infile.write(str(train_matrix[i][j]))
            infile.write(",")
        infile.write("\n")    
    classy_list = training['class']
    #cost_function(train_matrix,classy_list)
    #print('calling predict_LR')
    predict_LR('TestSet\Final_project_test.txt',train_matrix,count_variables,classy_list)
    
def main():
    train_read_files('TrainingSet\Final_project_train.txt')
if __name__ == "__main__" : main()