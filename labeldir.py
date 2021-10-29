import os
img_dir=os.listdir('./img_640')
traintxt=open("train.txt",'w')
for img_name in img_dir:
    path=os.path.join(os.getcwd(),'img_640',img_name)+'\n'
    traintxt.write(path)
    #print(path)
traintxt.close()    