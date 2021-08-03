/*
 *Modifications Copyright 2017, ING Wholesale Banking Advanced Analytics Team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */



import pandas as pd

memberlist=pd.read_excel(r'C:\Users\user\Desktop\Combined member list.xlsx',header=None,usecols=[0,2])  #in original dataset, column 0 is member's name, column 2 is the year of the member's registration
attendancelist=pd.read_excel(r'C:\Users\user\Desktop\Combined attendance list of events.xlsx',header=None,usecols=[0]) #in original dataset, column 0 is the member's name

#In this case,memberlist is the membership database,attendancelist is a combined attendance list of all events held in the year
#The final output returns the number of events attended by each member in the year (separated into 2 files: activememberlist.xlsx and nonactivememberlist.xlsx)
#The final output also returns the names in attendancelist who are not recorded in the membership database (nonmemberlist.xlsx)


#match all names in attendancelist (data to looked for) with the names in memberlist (the database)
#the data in member_matchingcolumn and attendance_matchingcolumn used here should be all in UPPERCASE/lowercase

member_matchingcolumn=0             #use which column in memberlist to match the columns in attendancelist
attendance_matchingcolumn=0         #use which column in attendancelist to match the column in memberlist



############################################################################
#The following part is direct matching to match absolutely identical data

attendancecountcolumn=len(memberlist.columns)
memberlist.insert(attendancecountcolumn,'Attendancecount',0)

member_matchingcolumn_index=memberlist.columns.get_loc(member_matchingcolumn)


activemember=[]
nonmember=[]

for i in range (len(attendancelist)):
    what=attendancelist.iloc[i][attendance_matchingcolumn]
    index=memberlist.index[(memberlist[member_matchingcolumn]==what)]
    member=memberlist.iloc[index].to_numpy()

    if len(member)>0:
        
        index=memberlist.index[(memberlist[member_matchingcolumn]==member[0][member_matchingcolumn_index])][0]
        memberlist.iat[index,attendancecountcolumn]+=1
    
    else:
        n_member=attendancelist.iloc[i].to_numpy()
        nonmember.append(n_member)

if len(attendancelist.columns)==1:
    nonmemberlist=pd.DataFrame(nonmember).melt()['value']
    nonmemberlist=nonmemberlist.to_frame(attendance_matchingcolumn)

else:
    nonmemberlist=pd.DataFrame(nonmember)

##############################################################################
#The following part is modified from sparse_dot_topn to match inconsistent names
#This part could be skipped and directly go to the exportation session if all the names are consistent/the data used for matching is numbers (this part only works for strings)

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct
import numpy as np
from tkinter import Tk,Button,Frame
from pandastable import Table, TableModel

root=Tk()
root.geometry('1280x680')
root.config(bg='white')
root.title('Name Matching')
frame=Frame(root)
frame.pack(fill='both',expand=True)


pd.set_option('display.expand_frame_repr', False)

def ngrams(string,n=3):
    string=re.sub('[/]|\sBD',r'',string)
    ngrams=zip(*[string[i:] for i in range (n)])
    return [''.join(ngram) for ngram in ngrams]

#awesome_cossim_top is modified from awesome_cossim_topn of its original library
def awesome_cossim_top(A,B,ntop,lower_bound=0):
    A=A.tocsr()   
    B=B.tocsr()

    M,_=A.shape
    _,N=B.shape
    
    idx_dtype=np.int32
    
    nnz_max=M*ntop
    
    indptr=np.zeros(M+1,dtype=idx_dtype)   
    indices=np.zeros(nnz_max,dtype=idx_dtype)   
    data=np.zeros(nnz_max,dtype=A.dtype)   
    
    ct.sparse_dot_topn(
        M,N,
        np.asarray(A.indptr,dtype=idx_dtype),
        np.asarray(A.indices,dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr,dtype=idx_dtype),
        np.asarray(B.indices,dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr,indices,data
    )
    return csr_matrix((data,indices,indptr),shape=(M,N))

def get_matches_df(sparse_matrix,name_vector,originaldatalist):   #originaldatalist=memberlist_sliced, name_vector=processingdata
    non_zeros=sparse_matrix.nonzero()
    sparserows=non_zeros[0]
    sparsecols=non_zeros[1]
    
    nr_matches=sparsecols.size
    
    left_side=np.empty([nr_matches],dtype=object)
    right_side=np.empty([nr_matches],dtype=object)
    similarity=np.zeros(nr_matches)
    
    left_side_index=np.empty([nr_matches],dtype=object)
    right_side_index=np.empty([nr_matches],dtype=object)

    
    for index in range(0,nr_matches):
        if sparserows[index]>=len(originaldatalist) and sparsecols[index]<len(originaldatalist):
            
            left_side_index[index]=sparserows[index]-len(originaldatalist)
            left_side[index]=name_vector[sparserows[index]]
            right_side_index[index]=sparsecols[index]
            right_side[index]=name_vector[sparsecols[index]]
            similarity[index]=sparse_matrix.data[index]
    
    searchresult=pd.DataFrame({'Index of data to look for':left_side_index,'Data to look for':left_side,'Index of possible matches':right_side_index,'Possible matches':right_side,'Similarity':similarity}).dropna()
    searchresult.reset_index(drop=True,inplace=True)
    return  searchresult

memberlist_sliced=pd.DataFrame(memberlist[member_matchingcolumn])

memberlist_sliced.columns=['Data for processing']


attendancelist_unidentified=pd.DataFrame(nonmemberlist[attendance_matchingcolumn])
attendancelist_unidentified.columns=['Data for processing']

processingdata=pd.concat([memberlist_sliced,attendancelist_unidentified],axis=0,ignore_index=True)['Data for processing']   #concat and slice (select 'data for processing column only')

vectorizer=TfidfVectorizer(min_df=1,analyzer=ngrams)
processingdata_idf_matrix=vectorizer.fit_transform(processingdata)
matches=awesome_cossim_top(processingdata_idf_matrix,processingdata_idf_matrix.transpose(),10,0.4)
#matches will similarity below 0.4 will be filtered out directly
#similarity of 1==exact match, similarity of 0==not matching at all

matches_df=get_matches_df(matches,processingdata,memberlist_sliced)
matches_df=matches_df.sort_values(['Index of data to look for','Similarity'],ascending=[True,False])


datalookedfor=np.unique(matches_df['Index of data to look for'].to_numpy())
#datalookedfor is a unique list of index of all names to look for

datalookedfor_value_track=pd.concat([matches_df.index.to_frame(),matches_df.iloc[:,[0,2]]],axis=1).to_numpy()
#datalookedfor_value_track column 0 is the index of original list (the index which rowclicked returns), column 1 is the value of unique list, this is to find the index of that cell in the unique list, column 2 is the index of possible matches


showmatches=matches_df
showmatches.columns=['No.','Look for','Index of matches','Potential matches','Similarity']


clickcount=np.zeros(len(showmatches))

datamatched=np.full((len(datalookedfor),2),None)
#column 0 of datamatched is the the index of possible matches,column 1 is the index in original dataset which is equal to rowclicked
#create a len(datalookedfor) list with all cells filled with None


pt=Table(frame, dataframe=showmatches)
pt.show()


def handle_left_click(event):
    global clickcount,datamatched
    #Handle mouse left click
    rowclicked = pt.get_row_clicked(event)
    
    
    datalookedfor_value=datalookedfor_value_track[rowclicked,1]
    #find value of the cell
    datalookedfor_index=np.where(datalookedfor==datalookedfor_value)[0][0]
    #use its value to find its index in the unique list
    datalookedfor_cell=datalookedfor[datalookedfor_index]
    #the position of that cell in the unique list 
    
 
    if clickcount[rowclicked]==0:
        if datamatched[datalookedfor_index,1]!=None:
            pt.setRowColors(rows=[datamatched[datalookedfor_index,1]], clr='#F4F4F3',cols='all')
            clickcount[datamatched[datalookedfor_index,1]]=0
        datamatched[datalookedfor_index,0]=datalookedfor_value_track[rowclicked,2]
        datamatched[datalookedfor_index,1]=rowclicked
        pt.rowselectedcolor = 'pale green'
        pt.setRowColors(rows=[rowclicked], clr='pale green',cols='all') #cols=[0,1]
        clickcount[rowclicked]=1
        
    else:
        pt.rowselectedcolor = '#F4F4F3'
        datamatched[datalookedfor_index,0]=None
        datamatched[datalookedfor_index,1]=None
        pt.setRowColors(rows=[rowclicked], clr='#F4F4F3',cols='all') #cols=[0,1]
        clickcount[rowclicked]=0
    
    
    pt.setSelectedRow(rowclicked)
    pt.redraw()


pt.bind('<Button-1>',handle_left_click)
pt.rowheader.bind('<Button-1>',handle_left_click)


def exportdata():
    global datalookedfor,datamatched,nonmemberlist,memberlist,attendancecountcolumn
    for i in range(len(datalookedfor)):
        index=datamatched[i,0]

        if index!=None:
            nonmemberlist.drop(labels=[datalookedfor[i]],axis=0,inplace=True)
            memberlist.iat[index,attendancecountcolumn]+=1
            
    nonmemberlist.reset_index(drop=True,inplace=True)
    root.destroy()
    

confirm=Button(text='Confirm',bg='#303030',fg='pale green',command=exportdata)
confirm.place(x=1150,y=600,anchor='se')


root.mainloop()
#select only names that you think match the potential matches, unselected names will be treated as non-members directly


#############################################################################
#The following part is exportation of output files

nonmemberlist.columns=['Name']
memberlist.columns=['Name','Year','No. of Events Attended in the Year']

nonactivememberlist=memberlist.loc[memberlist['No. of Events Attended in the Year'] == 0]
activememberlist=memberlist.loc[memberlist['No. of Events Attended in the Year'] != 0]


print(nonactivememberlist)
print('\n')
print(activememberlist)
print('\n')
print(nonmemberlist)

nonmemberlist.to_excel(r'C:\Users\user\Desktop\List of non-members.xlsx',index=False)
nonactivememberlist.to_excel(r'C:\Users\user\Desktop\List of non-active members.xlsx',index=False)
activememberlist.to_excel(r'C:\Users\user\Desktop\List of active members.xlsx',index=False)




