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


    
  
def numpy_matrix(panda_df,count_variables,row_count):
    print('inside np_matrix')
    desc_list = panda_df['desc'].tolist()
    
    np_matrix = np.zeros((row_count,len(count_variables)))
    #print('the np_matrix is',np.shape(np_matrix))
    count_variables_list = list(count_variables)
    line_number = 0
    for line in desc_list:
        print(line_number)
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
    

def accuracy(final_list,test):
    test_class_list = test['class'].tolist()
    test_class_pd = pd.DataFrame(test_class_list)
    test_class_pd = test_class_pd.replace('student',0)
    test_class_pd = test_class_pd.replace('faculty',1)
    test_class_pd = test_class_pd.replace('project',2)
    test_class_pd = test_class_pd.replace('course',3)
    final_list_pd = pd.DataFrame(final_list)
    final_list_pd = final_list_pd.replace('student',0)
    final_list_pd = final_list_pd.replace('faculty',1)
    final_list_pd = final_list_pd.replace('project',2)
    final_list_pd = final_list_pd.replace('course',3)
    test_class_np = np.array(test_class_pd)
    final_list_np = np.array(final_list_pd)
    print("the accuracy is",np.mean(test_class_np == final_list_np)*100)


def NaiveBayes(class_list,variables_counter,path):
    test = pd.read_csv(path,sep='\t',names=['num','class','desc'],header = None)
    final_list = []
    desc_list = test['desc'].tolist()
    test_class = test['class']
    total_train_files = sum(class_count.values())
    i = 0
    for line in desc_list:
        class_prob.clear()

        if type(line) is not str:
            continue
        record = line.split()
        for word in record:
            for classes in class_list:
                prob_word_in_class = 0.0
                class_counter = complete_dict[classes]
                class_desc_overall_count = sum(class_counter.values())
                class_word_unique_count = len(variables_counter)
                word_count = class_counter.get(word,0)
                prob_word_in_class = ( ((math.log2((word_count+1)) - math.log2((class_desc_overall_count + class_word_unique_count)))))
                if class_prob.get(classes,0) != 0:
                    class_prob[classes] =  class_prob[classes] + prob_word_in_class
                else:
                    class_prob[classes] =  prob_word_in_class
        class_prob[classes] = class_prob[classes] + math.log2(class_count[classes]/total_train_files)
        #print(class_prob,max(class_prob,key=class_prob.get))
        final_list.append(max(class_prob,key=class_prob.get))            
    accuracy(final_list,test)       
                
                
                
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
    
    NaiveBayes(class_list,count_variables,'TestSet\Final_project_test.txt')
    #train_np = np.zeros((sum(class_count.values()),len(count_variables)))
    #print('the shape of train_np is',np.shape(train_np))
    #train_matrix = np.zeros((sum(class_count.values()),len(count_variables)))
    '''train_matrix = numpy_matrix(training,count_variables,sum(class_count.values()))
    #print('the train_npy is',train_matrix)
    #print(np.count_nonzero(train_matrix))
    '''
    #Add X0 column in train_matrix
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
    #cost_function(train_matrix,classy_list)'''
    
    
    
def main():
    train_read_files('TrainingSet\Final_project_train.txt')
if __name__ == "__main__" : main()