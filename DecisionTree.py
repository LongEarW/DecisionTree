# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 18:36:47 2019

@author: Wenqing
"""


class DecTree:
    
    def __init__(self,dic_data, testdata, dim, dep):
        self.maxdep = dep
        self.dic_data = dic_data #label: [[f1,f2,..],...]
        self.test_data = testdata
        self.dim = dim
        self.rules = {} #dep:[(tag,feature,split),...]; tag= 0,01,02,012
    
    def splSub_fun(self,cand_split, feature, dic_loc):
        '''calculate entropy'''
        left_sub = {}
        right_sub = {}
        for label in dic_loc:
            right_sub[label]=[]
            left_sub[label]=[]
            for indx in dic_loc[label]:
                if self.dic_data[label][indx][feature] > cand_split:
                    right_sub[label].append(indx)
                else:
                    left_sub[label].append(indx)

        return(left_sub, right_sub)
    
    def getMajLab(self, dic):
        '''get the label with maximum frequency'''
        iter_lab = list(dic.keys())
        iter_lab.sort()
        count = 0
        maj_lab = -1
        for label in iter_lab:
            can_count = len(dic[label])
            if count< can_count:
                count = can_count
                maj_lab = label
        return(maj_lab)
        
    def split_fun(self,dic_loc,dep,tag):
        '''look for split'''
        min_entr = 1
        min_feature = -1
        split = -1
        left_sub = {}
        right_sub = {}
        iter_label = list(dic_loc.keys())
        iter_label.sort()
        #try through all split for each feature
        for feature in range(self.dim):
            #get candidate split
            sample_col = []
            for label in iter_label:
                for indx in dic_loc[label]:
                    sample_col.append(self.dic_data[label][indx][feature])

            sample_col_uni = list(set(sample_col))
            sample_col_uni.sort()
            cand_split_ls = list((sample_col_uni[i]+sample_col_uni[i+1])/2\
                              for i in range(len(sample_col_uni)-1))
           
            #get cmf for each lable
            dic_cmf = {}
            for label in iter_label:
                dic_cmf[label] = {}
                sample_col_temp = list(self.dic_data[label][indx][feature] for indx in dic_loc[label])
                sample_col_temp.sort()
                cmf = 0
                for can_split in cand_split_ls:
                    if cmf==len(sample_col_temp):
                        dic_cmf[label][can_split] = cmf
                        continue                    
                    while sample_col_temp[cmf] <= can_split:
                        cmf+=1
                        if cmf==len(sample_col_temp):
                            break
                    dic_cmf[label][can_split] = cmf

                    
            for can_split in cand_split_ls:
                count_label_l = []
                count_label_r = []
                for label in dic_cmf:
                    count_t = len(dic_loc[label])
                    cmf = dic_cmf[label][can_split]
                    count_label_l.append(cmf)
                    count_label_r.append(count_t-cmf)
                gini_l = 1-sum(list((c/sum(count_label_l))**2 for c in count_label_l))
                gini_r = 1-sum(list((c/sum(count_label_r))**2 for c in count_label_r))
                entro = (sum(count_label_l)*gini_l+sum(count_label_r)*gini_r)/len(sample_col)
                if entro < min_entr:
                    min_entr = entro
                    min_feature = feature
                    split = can_split
                    min_gl = gini_l
                    min_gr = gini_r
        #generate subset
        left_sub, right_sub = self.splSub_fun(split, min_feature, dic_loc)
        if  min_gl ==0:
            left_pure = 1
        else:
            left_pure = 0
        if  min_gr ==0:
            right_pure = 1
        else:
            right_pure = 0                    

        #recursive
        tag_l = tag.copy()
        tag_r = tag.copy()
        tag_l.append(1)
        tag_r.append(2)
        if  min_feature!=-1:
            if dep not in self.rules:
                self.rules[dep] = []
            self.rules[dep].append((tag,min_feature,split))
            
        if dep < self.maxdep and min_feature!=-1:
            if left_pure==0:
                self.split_fun(left_sub, dep+1,tag_l)
            else:
                #leaf node
                major_label = self.getMajLab(left_sub)
                if dep+1 not in self.rules:
                    self.rules[dep+1] = []
                self.rules[dep+1].append((tag_l,major_label))
            if right_pure==0:
                self.split_fun(right_sub, dep+1,tag_r)
            else:
                #leaf node
                if dep+1 not in self.rules:
                    self.rules[dep+1] = []                
                major_label = self.getMajLab(right_sub)
                self.rules[dep+1].append((tag_r,major_label))                
        elif dep==self.maxdep:
            #reach maximum depth
            if dep+1 not in self.rules:
                self.rules[dep+1] = []
            major_label_l = self.getMajLab(left_sub)
            major_label_r = self.getMajLab(right_sub)
            self.rules[dep+1].append((tag_l,major_label_l))
            self.rules[dep+1].append((tag_r,major_label_r))
            
    def predict_fun(self):
        '''predict the test data'''
        result = []
        for sample in self.test_data:
            dep = 1
            position = [0]
            while dep<=self.maxdep+1:
                for rule in self.rules[dep]:
                    if rule[0]== position:
                        if len(rule)==2:
                            #leaf
                            result.append(rule[1])
                            break
                        else:
                            #root
                            split = rule[2]
                            feature = rule[1]
                            if sample[feature] > split:
                                position.append(2)
                                break
                            else:
                                position.append(1)
                                break
                dep+=1
        return(result)
        
    def bldTree(self):
        '''build decision tree'''
        dic_loc = {}
        dep = 1
        tag = [0]
        for label in self.dic_data.keys():
            dic_loc[label] = list(range(len(self.dic_data[label])))
        self.split_fun(dic_loc, dep, tag)
        
        
        
if __name__ == '__main__':
    
    dep = 2
    k = 3
    line_ls = []
    
    while True:
        line = input()
        if not line:
            break
        line_ls.append(line)
    dic_data, testdata, dim = PrePross(line_ls) 

    tree_obj = DecTree(dic_data, testdata, dim, dep)
    tree_obj.bldTree()
    result_tree = tree_obj.predict_fun()
    for result in result_tree:
        print(result)
