import torch
import random

####initial####
w1=torch.tensor(0.0543,requires_grad=True)
w2=torch.tensor(0.0579,requires_grad=True)
w3=torch.tensor(-0.0291,requires_grad=True)
w4=torch.tensor(0.0999,requires_grad=True)
w5=torch.tensor(0.0801,requires_grad=True)
w6=torch.tensor(-0.0605,requires_grad=True)
b1=torch.tensor(-0.0703,requires_grad=True)
b2=torch.tensor(-0.0939,requires_grad=True)
b3=torch.tensor(-0.0109,requires_grad=True)
alpha=0.6

###fuction###
fx=lambda x:1/(1+torch.exp(-x))
lossfc=lambda x,y:torch.absolute(x-y)# x is output, y is target


def computeOut(x1,x2):
    y1 = fx(x1 * w1 + x2 * w3 +b1)
    y2 = fx(x1 * w2  + x2 * w4 + b2)
    out = fx(w5 * y1  + y2 * w6 + b3)
    return out
def zeroGrad(weights:list):
    for item in weights:
        item.grad=None
def requiresGrad(weights:list):
    for i in weights:
        i.requires_grad=True
times=1
def trainepoch(x1,x2,target):
    ##compute
    global w1,w2,w3,w4,w5,w6,b1,b2,b3,times
    x1=torch.tensor(x1)
    x2=torch.tensor(x2)

    requiresGrad([w1,w2,w3,w4,w5,w6,b1,b2,b3])

    out=computeOut(x1,x2)
    target=float(target)


    ###update
    loss=lossfc(out,target)
    loss.backward()

    with torch.no_grad():
        w1 = w1 - alpha * w1.grad
        w2 = w2 - alpha * w2.grad
        w3 = w3 - alpha * w3.grad
        w4 = w4 - alpha * w4.grad
        w5 = w5 - alpha * w5.grad
        w6 = w6 - alpha * w6.grad
        b1 = b1 - alpha * b1.grad
        b2 = b2 - alpha * b2.grad
        b3 = b3 - alpha * b3.grad

        out_update=computeOut(x1,x2)
        error=torch.absolute(out_update-target)

    zeroGrad([w1, w2, w3, w4, w5, w6, b1, b2, b3])


    print("{},{},{},out:{},out_update:{},loss:{},error:{}     {}".format(x1,x2,target,"%.6f"%out,"%.6f"%out_update,"%.6f"%loss,"%.4f"%error,times))
    times+=1
    return lossfc(out_update,target),error
sets=[[0,0,0],[0,1,1],[1,0,1],[1,1,0]]
# sets=[[1,1,0]]
for i in range(36000):
    print(i+1,end=',')
    totalloss=0
    totalerror=0
    random.shuffle(sets)
    errors=[]
    for index,set in enumerate(sets):
        x1,x2,y=set
        loss,error=trainepoch(x1,x2,y)
        errors.append(error)
        totalloss+=loss
        totalerror+=error
    if max(errors)<0.008:
        print("w1:{},   w2:{},   w3:{},   w4:{},   w5:{},   w6:{},   b1:{},   b2:{},   b3:{}".
              format(w1,w2,w3,w4,w5,w6,b1,b2,b3))


        print('训练结束')
        break
    x1,x2,y=sets[errors.index(max(errors))]
    for i in range(2):
        loss,error=trainepoch(x1,x2,y)
    # print(totalloss.item()/4,totalerror.item()/4)