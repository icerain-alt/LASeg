import os
from sklearn.model_selection import train_test_split

data_path = 'E:/data/LASet'
names = os.listdir(os.path.join(data_path,'origin'))
train_ids,test_ids = train_test_split(names,test_size=0.2,random_state=367)
with open(os.path.join(data_path,'train.list'),'w') as f:
    f.write('\n'.join(train_ids))
with open(os.path.join(data_path,'test.list'),'w') as f:
    f.write('\n'.join(test_ids))
print(len(names),len(train_ids),len(test_ids))