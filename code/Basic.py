import numpy as np  
import matplotlib.pyplot as plt
import tqdm
from tqdm import tqdm


#L <=W


p1 = 0.2
p12 = 0.2
p2 = 0.25
eta1 = 0.2
eta2 = 0.1
u1 = 0.1
u2 = 0.05
Horizon = 30


window = 5

S1 = np.zeros([Horizon])
I1 = np.zeros([Horizon])
R1 = np.zeros([Horizon])
A1 = np.zeros([Horizon])
S2 = np.zeros([Horizon])
I2 = np.zeros([Horizon])
R2 = np.zeros([Horizon])
A2 = np.zeros([Horizon])

S1[0] = 190
S2[0] = 190
A1[0] = 10
A2[0] = 10
I1[0] = 0
I2[0] = 6
R1[0] = 0
R2[0] = 0
pop = 200

v1 = np.array([0.2,0.3,0.4]) 
v2 = np.array([0.1,0.2,0.3])
L1frac = np.array([0.3,0.5,0.7,1])
L2frac = np.array([0.3,0.5,0.7,1])
L12frac = np.array([0.3,0.5,0.7,1])
Benef1 = 3
Benef12 = 2


def bounds(Z1,Z2,Ls,controls,window):
    l2 = Ls
    v2 = controls
    w1 = Z1[0] +  Z1[2] + Z1[3]
    w2 = Z2[0] +  Z2[2] + Z2[3]
    maxx  = [Z1[0],Z1[1],Z1[2],Z2[0],Z2[1],Z2[2]] ## S1,I1,A1,S2,I2,A2
    minn  = [Z1[0],Z1[1],Z1[2],Z2[0],Z2[1],Z2[2]]
    [Curr1,Curr2] = [Z1,Z2]    
    step = []
    
    for i in range(len(v1)):
        for j in range(len(L1frac)):
            for mm in range(len(L12frac)):
                [Curr1,Curr2] = [Z1,Z2] 
                for p in range(window):
                    s1n,i1n,a1n,r1n,s2n,i2n,a2n,r2n = update(Curr1,Curr2,[L1frac[j]*w1,l2,L12frac[mm]*w1],[v1[i],v2])
                    Curr1 = [s1n,i1n,a1n,r1n] 
                    Curr2 = [s2n,i2n,a2n,r2n] 
                    w1 = s1n + a1n + r1n
                    w2 = s2n + a2n + r2n        
                    step = [s1n,i1n,a1n,s2n,i2n,a2n]
                    

                    for k in range(6):
                        if(step[k]>maxx[k]):
                            maxx[k] = step[k]
                    for k in range(6):
                        if(step[k]<minn[k]):
                            minn[k] = step[k]
    print("$$$$$$$$$$$$4")                    
    print(maxx)
    print(minn)
    print("GGGGGGGGGGGGGg")   
    return maxx, minn            





def controls_update(Z1,Z2, boundings,Ls, controls):
    s1,i1,a1,r1 = Z1
    s2,i2,a2,r2 = Z2
    max_l, min_l = boundings
    
    max_l = np.asarray(max_l,dtype = int)
    min_l = np.asarray(min_l,dtype = int)
    vals = np.asarray(max_l - min_l + 1,dtype=int)

    l2 = Ls
    v2 = controls
 
    
    V = 1e10*np.ones([window,vals[0],vals[1],vals[2],vals[3],vals[4],vals[5]])
    for i in range(max_l[1],min_l[1]-1,-1):
        V[window-1,:, i - min_l[1],:,:,:,:] = i #s,i,a,s2,i2,a2


    

    for ww in tqdm(range(window-2,-1,-1)):
        for i in tqdm(range(max_l[0],min_l[0]-1,-1)): #S
            for j in tqdm(range(max_l[1]-1,min_l[1]-1,-1)): #I
                for k in range(max_l[2],min_l[2]-1,-1): #A
                    for i2 in range(max_l[3],min_l[3]-1,-1): #S2
                        for j2 in range(max_l[4],min_l[4]-1,-1): #I2
                            for k2 in range(max_l[5],min_l[5]-1,-1): #A2
                                
                                opt = 1e10
                                for x in range(len(L1frac)):
                                    for y in range(len(L12frac)):
                                        for z in range(len(v1)):
                                            Z1_temp = [i,j,k,0]
                                            Z2_temp = [i2,j2,k2,0]
                                            w1 = pop - Z1_temp[1]
                                            w2 = pop - Z2_temp[1]  
                                            
                                            s1n,i1n,a1n,r1n,s2n,i2n,a2n,r2n = update(Z1_temp,Z2_temp,[L1frac[x]*w1,l2,L12frac[y]*w1],[v1[z],v2])
                                            
                                                
                                            v_arr = [s1n,i1n,a1n,s2n,i2n,a2n] - min_l
                                            boo = ([s1n,i1n,a1n,s2n,i2n,a2n]>=min_l).all() and ([s1n,i1n,a1n,s2n,i2n,a2n]<=max_l).all()
                                            if(boo):
                                                temp = max(j, V[ww+1, v_arr[0],v_arr[1],v_arr[2],v_arr[3],v_arr[4],v_arr[5]])
                                                if(temp<opt):
                                                    opt = temp


                                            # if((Z1_temp[:3]==Z1[:3])and (Z2_temp[:3]==Z2[:3]) and (ww==0)):
                                                # print(opt, j,i1n,temp)
                                                # print("OK")
                                                



                                final_arr = [i,j,k,i2,j2,k2] - min_l
                                V[ww,final_arr[0],final_arr[1],final_arr[2],final_arr[3],final_arr[4],final_arr[5]] = opt
                                # if(ww==1 or ww==0):
                                    # print("hello")
                                    # print(V[ww,final_arr[0],final_arr[1],final_arr[2],final_arr[3],final_arr[4],final_arr[5]])
                                    # print(j)
                                # print(V[window-ww-1,final_arr[0],final_arr[1],final_arr[2],final_arr[3],final_arr[4],final_arr[5]])
    # print(Z1[0],Z1[1],Z1[2],Z2[0],Z2[1],Z2[2])
    fin = np.asarray([Z1[0],Z1[1],Z1[2],Z2[0],Z2[1],Z2[2]]  - min_l,dtype = int)
    # print(fin[0],fin[1],fin[2],fin[3],fin[4],fin[5])   
    # print("optimality",V[0,fin[0],fin[1],fin[2],fin[3],fin[4],fin[5]])
    # print("lllllllllll")
    optimal_value = V[0,fin[0],fin[1],fin[2],fin[3],fin[4],fin[5]]
    V = np.array(V)
    # # print(V[])
    # print(V[np.where(V>0)])
    count1 = 0
    count2 = 0
    print("OPTI VAL",optimal_value)
   
    for x in tqdm(range(len(L1frac))):
        for y in range(len(L12frac)):
            for z in range(len(v1)):    
                Z1_temp = Z1
                Z2_temp = Z2
                w1 = pop - Z1_temp[1]
                w2 = pop - Z2_temp[1]  
                s1n,i1n,a1n,r1n,s2n,i2n,a2n,r2n = update(Z1_temp,Z2_temp,[L1frac[x]*w1,l2,L12frac[y]*w1],[v1[z],v2])
                v_arr = np.asarray([s1n,i1n,a1n,s2n,i2n,a2n]-min_l,dtype = int)
                count1 = count1+1
                # print(i1n)
                print("MATCHER",V[1, v_arr[0],v_arr[1],v_arr[2],v_arr[3],v_arr[4],v_arr[5]])
                # print("XXXXX")
                # print(v_arr)
                if(optimal_value==V[1, v_arr[0],v_arr[1],v_arr[2],v_arr[3],v_arr[4],v_arr[5]]):
                    # print(L1frac[x],L12frac[y],v1[z])
                    return L1frac[x],L12frac[y],v1[z]
                    break
                    # count2 = count2+1
    print("No optimal found")                
    return L1frac[0],L12frac[0],v1[0]
    # print(max_l[1]-1,min_l[1]-1,"LLLLLLL")
    # print(count1,count2)

#Z = [S,I,A,R]
def update(Z1,Z2,Lcurr,controls):
    v1t,v2t = controls
    l1,l2,l12 = Lcurr
    s1,i1,a1,r1 = Z1
    s2,i2,a2,r2 = Z2
    w1 = pop - i1
    w2 = pop - i2  
    s1n = s1 - int((s1/w1)*((a1/w1)*p1*l1 + (a2/w2)*p12*l12))
    s2n = s2 - int((s2/w2)*((a2/w2)*p2*l2 + (a1/w1)*p12*l12))
    a1n = a1 + int((s1/w1)*((a1/w1)*p1*l1 + (a2/w2)*p12*l12)) - round(v1t*a1) - round(u1*a1)
    a2n = a2 + int((s2/w2)*((a2/w2)*p2*l2 + (a1/w1)*p12*l12)) - round(v2t*a2) - round(u2*a2)
    i1n = i1 + round(v1t*a1) - round(eta1*i1)
    i2n = i2 + round(v2t*a2) - round(eta2*i2)
    r1n = r1 + round(u1*a1) + round(eta1*i1)
    r2n = r2 + round(u2*a2) + round(eta2*i2)
    return s1n,i1n,a1n,r1n,s2n,i2n,a2n,r2n



for i in range(1,Horizon):
    Z1 = [S1[i-1],I1[i-1],A1[i-1],R1[i-1]]
    Z2 = [S2[i-1],I2[i-1],A2[i-1],R2[i-1]]
    W1 = S1[i-1] + A1[i-1] + R1[i-1] 
    W2 = S2[i-1] + A2[i-1] + R2[i-1] 
    L1 = 0.5*W1
    L2 = 0.5*W2
    L12 = 0.3*min(W1,W2)
    # Lcurr = [L1,L2,L12]
    # controls = [v1[0],v2[0]]
    # controls = [v1[i-1],v2[i-1]]
    max_l,min_l = bounds(Z1,Z2,L2,v2[0],window = 5)
    l1n,l12n,v1n = controls_update(Z1,Z2, [max_l,min_l],L2, v2[0])
    controls = [v1n,v2[0]]
    Lcurr = [l1n*W1,L2,l12n*W1]
    S1[i],I1[i],A1[i],R1[i],S2[i],I2[i],A2[i],R2[i] = update(Z1,Z2,Lcurr,controls)
    print("Time Step ",i,"Values","S1= ",S1[i],"I1= ",I1[i],"A1= ",A1[i],"R1= ",R1[i],"S2= ",S2[i],"I2= ",I2[i],"A2= ",A2[i],"R2= ",R2[i],"\n")
    print("Total",S1[i]+I1[i]+A1[i]+R1[i],S2[i]+I2[i]+A2[i]+R2[i],"\n")


print(S1)
print(S2)
print(I1)
print(I2)
print(A1)
print(A2)
print(R1)
print(R2)

# S1 = [190, 189, 189, 189, 189, 189, 189, 187, 187, 187, 185, 185, 185, 183, 183, 183, 183., 181., 181., 181., 179., 179., 179., 177., 177., 177., 175., 175., 175., 175., 175., 175., 175., 175., 175., 175., 175., 175., 175., 175., 175., 175., 175., 175., 175.,175., 175., 175., 175., 175., 175., 175.]
# S2 = [190, 189, 188, 187, 186, 185, 184, 183, 182, 181, 180, 179, 178, 177, 176, 176., 175., 174., 173., 172., 171., 170., 169., 168., 167., 166., 165., 164., 163., 162., 161., 160., 159., 158., 157., 156., 155., 154., 153., 152., 151., 150., 149., 148., 148.,148.,148.,148.,148.,148., 148, 148]
# I1 = [0, 3, 3, 3, 3, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3., 3., 2., 3., 3., 2., 3., 3., 2., 3., 3., 2., 3., 3., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,2,2,2,2,2,2,2]
# I2 = [0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,6,6,6,6,5,5,5]
# A1 = [10, 7, 5, 4, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 2., 1., 3., 2., 1., 3., 2., 1., 3., 2., 1., 3., 2., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,1,1,1,1,1,1,1]
# A2 = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10., 10., 10. ,10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.,  9., 8 ,7,6,5,5,5,5]
# R1 = [0, 1, 3, 4, 5, 6, 7, 8, 8, 9, 10, 10, 11, 12, 12, 12., 13., 14., 14., 15., 16., 16., 17., 18., 18., 19., 20., 20., 21., 22., 22., 22., 22., 22., 22., 22., 22., 22., 22., 22., 22., 22., 22., 22., 22.,22,22,22,22,22,22,22]
# R2 = [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8.,  9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19. ,20., 21., 22., 23. ,24. ,25. ,26. ,27. ,28. ,29., 30., 31., 32. ,33. ,34. ,35. ,36. ,37.,38,39,40,41,42,42,42]





plt.plot(S1, label="S1")
plt.legend()
plt.savefig("S")
plt.plot(S2, label="S2")
plt.legend()
plt.savefig("S")
plt.close()

plt.plot(I1, label="I1")
plt.legend()
plt.savefig("I")
plt.plot(I2, label="I2")
plt.legend()
plt.savefig("I")
plt.close()

plt.plot(A1, label="A1")
plt.legend()
plt.savefig("A")
plt.plot(A2, label="A2")
plt.legend()
plt.savefig("A") 
plt.close()

plt.plot(R1, label="R1")
plt.legend()
plt.savefig("R") 
plt.plot(R2, label="R2")
plt.legend()
plt.savefig("R")   
plt.close() 