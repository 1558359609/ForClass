import math

####initial####
w1=0.0543
w2=0.0579
w3=-0.0291
w4=0.0999
w5=0.0801
w6=-0.0605
b1=-0.0703
b2=-0.0939
b3=-0.0109
alpha=0.6

###fuction###
fx=lambda x:1/(1+math.exp(-x))
lossfc=lambda x,y:(x-y)**2# x is output, y is target
dfx=lambda x: math.exp(-x)/(1.0+math.exp(-x))**2
dloss=lambda x,y:2*(x-y)


def trainepoch(x1,x2,target):
    ##compute
    global w1,w2,w3,w4,w5,w6,b1,b2,b3
    y1=fx(x1*w1+b1)+fx(x2*w3+b1)
    y2=fx(x1*w2+b2)+fx(x2*w4+b2)
    out =fx(w5*y1+b3)+fx(y2*w6+b3)
    # print('out:{}'.format(out),end=',update:')
    ##gradient
    dy1_dw1=dfx(x1*w1+b1)*x1
    dy1_dw3=dfx(w3*x2+b1)*x2
    dy2_dw2=dfx(w2*x1+b2)*x1
    dy2_dw4=dfx(w4*x2+b2)*x2

    dy1_db1=dfx(x1*w1+b1)+dfx(x2*w3+b1)
    dy2_db2=dfx(x1*w2+b2)+dfx(x2*w4+b2)

    dloss_dout=dloss(out,target)
    dloss_dw1=dloss_dout*(dfx(w5*y1+b3)*dy1_dw1)
    dloss_dw2=dloss_dout*(dfx(w6*y2+b3)*dy2_dw2)
    dloss_dw3=dloss_dout*(dfx(w5*y1+b3)*dy1_dw3)
    dloss_dw4=dloss_dout*(dfx(w6*y2+b3)*dy2_dw4)
    dloss_dw5=dloss_dout*(dfx(w5*y1+b3)*y1)
    dloss_dw6=dloss_dout*(dfx(w6*y2+b3)*y2)

    dloss_db1=dloss_dout*(dfx(w5*y1+b3)*dy1_db1)
    dloss_db2=dloss_dout*(dfx(w6*y2+b3)*dy2_db2)
    dloss_db3=dloss_dout*(dfx(w5*y1+b3)+dfx(y2*w6+b3))

    ###update
    w1=w1-alpha*dloss_dw1
    w2=w2-alpha*dloss_dw2
    w3=w3-alpha*dloss_dw3
    w4=w4-alpha*dloss_dw4
    w5=w5-alpha*dloss_dw5
    w6=w6-alpha*dloss_dw6
    b1=b1-alpha*dloss_db1
    b2=b2-alpha*dloss_db2
    b3=b3-alpha*dloss_db3

    y1 = fx(x1 * w1 + b1) + fx(x2 * w3 + b1)
    y2 = fx(x1 * w2 + b2) + fx(x2 * w4 + b2)
    out_update = fx(w5 * y1 + b3) + fx(y2 * w6 + b3)

    print("{},{},{},out:{},out_update:{},loss:{}".format(x1,x2,target,"%.6f"%out,"%.6f"%out_update,"%.6f"%lossfc(out_update,target)))

# sets=[[0,0,0],[0,1,1],[1,0,1],[1,1,0]]
sets=[[0,1,1]]

for i in range(1):
    for set in sets:
        x1,x2,target=set
        trainepoch(x1,x2,target)

