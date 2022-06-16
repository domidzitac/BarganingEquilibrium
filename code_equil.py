import numpy as np
from scipy.linalg import norm,solve
#import pandas as pd
from ast import literal_eval

import pandas as pd

#Constants
q = 2
d = np.array([0.85, 0.85, 0.85])
F=320

#e's exponenet
exp=1

#type_e can be symmetric or asymmetric for the e values
type_e="asymmetric_e_values"

#for the interval
lower_bound = 10
upper_bound = 100
range_step=1

### Generate e's

if type_e == "symmetric_e_values":
  list_symmetric=[]
  for e in range(lower_bound,upper_bound+1):
    list_symmetric.append([e,e,e])
  print(list_symmetric)
  df_symmetric = pd.DataFrame()
  df_symmetric['e'] = list_symmetric
  df_symmetric
  df_symmetric.to_csv('symmetric_e_values_'+str(lower_bound)+'_'+str(upper_bound)+'.csv')

if type_e == "asymmetric_e_values":
  list_asymmetric=[]
  for e1 in range(lower_bound,upper_bound+1):
    for e2 in range(lower_bound,upper_bound+1):
      for e3 in range(lower_bound,upper_bound+1):
        list_asymmetric.append([e1,e2,e3])
  print(list_asymmetric)
  df_asymmetric = pd.DataFrame()
  df_asymmetric['e'] = list_asymmetric
  df_asymmetric
  df_asymmetric.to_csv('asymmetric_e_values_'+str(lower_bound)+'_'+str(upper_bound)+'.csv')

#Read the e's

if type_e == "symmetric_e_values":
  df = pd.read_csv('symmetric_e_values_'+str(lower_bound)+'_'+str(upper_bound)+'.csv')
elif type_e == "asymmetric_e_values":
  df = pd.read_csv('asymmetric_e_values_'+str(lower_bound)+'_'+str(upper_bound)+'.csv')

#Generate P values

def generate_p_values(e_list,exp):
  sum_of_e = 0
  p_list=[]
  #raise all the elements in the list to the given exponent
  e_list = [number ** exp for number in e_list]
  #print(e_list)
  sum_of_e = sum(e_list)
  for e in e_list:
    if sum_of_e > 0:
      p_list.append(e/sum_of_e)
    else:
      #The p function is a piecewise function, which takes the value 1/3 when e_i=0 for all i
      p_list.append(1/3)
  return p_list

df.e = df.e.apply(literal_eval)
e_list_of_lists = df.e
p=[]
#print(range(len(e_list_of_lists)))
for e in range(len(e_list_of_lists)):
  #print(generate_p_values(e_list_of_lists[e],exp))
  p.append(generate_p_values(e_list_of_lists[e],exp))

df['p'] = p

#Generate v values

def cbfq(q, p, d):
  if np.sum(p)!=1:
      print('\n')
      print('WARNING: recognition probabilities do not sum up to 1. Applying proportional adjustment.\n')
      p = p/np.sum(p)
  dp=d*p
  Lr=dp/(1-d)
  Hr=dp/(1-dp)
  Lv=(p/(1-d))
  overTh=np.sort(Lr)
  overTh=overTh[q-1]

  underTh=np.sort(Hr)
  underTh=underTh[q-1]

  Theta = np.union1d(Lr,Hr)
  print('\n')
  Theta = -np.sort(-Theta[(Theta >= underTh) & (Theta <= overTh)])
  Theta = np.hstack(((Theta[1:]).reshape(1,-1).T,(Theta[:-1]).reshape(1,-1).T))
  K = len(Theta[:,0])
  invd = 1/d
  invd[(d==0)] = 0
  b = np.ones((2,1))
  Theta = np.column_stack((Theta,((Theta[:,0]+Theta[:,-1])/2)))

  if overTh==underTh:
      h=(overTh <= Hr)
      l=(overTh > Lr)
      m=(overTh > Hr) & (overTh <= Lr)
      S=(1+sum(l*Lr))/(1+overTh*(q-sum(l)))
      r=overTh * S
  else:
      Kset=np.zeros((K,1))
      ratio=Theta[int(np.ceil(K/2)-1),2]
      h=(ratio <= Hr)
      l=(ratio > Lr)
      m=(ratio > Hr) & (ratio <= Lr)
      A = np.zeros((2,2))
      A[0,0]=1+sum(l*Lr)
      A[0,1]=q-sum(l)
      A[1,0]=sum(l*Lv)+sum(h*p)
      A[1,1]=sum(m*invd)+sum(h*p)
      x = solve(A,b)
      S=x[0]
      r=x[1]
      k=(np.zeros((K,1))==1)
      k[int(np.ceil(K/2))-1,0]=(1==1)
      while (np.any(r<(Theta[k-1,0]*S)) or np.any(r>(Theta[k-1,1]*S))):
          print('\n')
          Kset= np.all(Kset) or np.all(k)
          check = (r >= Theta[:,0]*S) & (r <= Theta[:,1]*S) & (~Kset)
          if np.all (r-Theta[k-1,0]*S>-np.finfo(float).eps) & np.any(r-Theta[k-1,1]*S<np.finfo(float).eps):
              print('WARNING: iterate outside piece of linearity likely due to machine precision.\n')
              break

          elif sum(check) ==0:

              ang=Theta[~(Kset-1),1]
              I=min(abs(ang-ratio))
              ratio=ang[int(I)]
              if ((r>overTh*S) or (r<underTh*S)):
                  print(f'Iteration {Kset} Iterate out of bounds. Restart at nearest piece of linearity.\n')
              else:
                  print(f'Iteration {Kset} A cycle was detected. Restart at nearest piece of linearity.\n')
          else:
              ang = Theta[check-1, 2]
              ratio = min(ang)
              print(f'Iteration {Kset}\n')

          h = (ratio<=Hr)
          l = (ratio>Lr)
          m = (ratio>Hr) & (ratio<=Lr)
          A[0,0] = 1+sum(l*Lr)
          A[0,1] = q-sum(l);
          A[1,0] = sum(l*Lv)+sum(h*p);
          A[1,1] = sum(m*invd)+sum(h*p);
          res = norm(A*np.vstack((S,r))-b);
          print(f'     Residual: {res}\n')
          x= solve(A,b)
          S=x[0]
          r=x[1]
          k=(ratio > Theta[:,0]) & (ratio<Theta[:,0])
      res=norm(A*np.vstack((S, r))-b)

      print('\n')
      print(f'Convergence achieved in iterations.  Residual is: {res} (may not be zero due to machine precision)\n')

  v = l*(Lv*S)+(m*invd)*r +(h*p)*(S+r)
  ri=l*(Lr*S)+m*r +h*(Hr*S)
  if r>0:
      mi=l*(1-p)+m*(invd+p*(S/r)+p)
  else:
      mi=l*(1-p)
  return v, S, r, ri, mi

p = df.p
v_list=[]
for pi in range(len(p)):
  #print(np.array(p[pi]))
  try:
    v, S, r, ri, mi = cbfq(q, np.array(p[pi]), d)
    v_list.append(v)
    print(pi)
  except:
    v_list.append([1/3,1/3,1/3])

df['v'] = v_list

#Test equation

def eql_equation(e,v,F):

  #DECLARE VARIABLES

  #define e1,e2,e3
  e1=e[0]
  e2=e[1]
  e3=e[2]

  #define v1,v2,v3
  v1=v[0]
  v2=v[1]
  v3=v[2]


  #declare max values for each e and v, as well as for the equation
  e1_max=-9999
  e2_max=-9999
  e3_max=-9999

  v1_max=-9999
  v2_max=-9999
  v3_max=-9999

  pi = -9999
  pi_max= -9999

  equil=True

  #if all e are the same, then do this step only for the first value of e
  if e1 == e2 and e1 == e3 and e2 == e3:

    for e1_i in range(lower_bound,upper_bound+1):
      try:
        #compute v_i using the new e vector
        v_i, S, r, ri, mi = cbfq(q, np.array(generate_p_values([e1_i,e2,e3],exp)), d)
        v1_i=v_i[0]
        v2_i=v_i[1]
        v3_i=v_i[2]

        if ( ((F* v1_i) + upper_bound -e1_i) > ((F*v1)+upper_bound -e1)) == True:

          #if this equation is true, there is no equilibrium
          equil=False

          #save the equation value
          pi = (F* v1_i) + upper_bound - e1_i


          #if there exists a value greater than the maximum profit value, change max
          if pi > pi_max:
            pi_max = pi
            e1_max = e1_i
            e2_max = e1_i
            e3_max = e1_i
            v1_max = v1_i
            v2_max = v1_i
            v3_max = v1_i
      except:
        print("Error")

  #else fix e's 2 at the time
  else:
    #fix e2 and e3
    for e1_i in range(lower_bound,upper_bound+1):
      try:
        #compute v_i using the new e vector
        v_i, S, r, ri, mi = cbfq(q, np.array(generate_p_values([e1_i,e2,e3],exp)), d)
        v1_i=v_i[0]
        v2_i=v_i[1]
        v3_i=v_i[2]

        if ( ((F* v1_i) + upper_bound -e1_i) > ((F*v1)+upper_bound -e1)) == True:

          #if this equation is true, there is no equilibrium
          equil=False

          #save the equation value
          pi = (F* v1_i) + upper_bound - e1_i


          #if there exists a value greater than the maximum profit value, change max
          if pi > pi_max:
            pi_max = pi
            e1_max = e1_i
            v1_max = v1_i
      except:
        print("Error")

    #reset pi and pi_max
    pi = -9999
    pi_max= -9999

    #fix e1 and e3
    for e2_i in range(lower_bound,upper_bound+1):
      try:
        #compute v_i using the new e vector
        v_i, S, r, ri, mi = cbfq(q, np.array(generate_p_values([e1,e2_i,e3],exp)), d)
        v1_i=v_i[0]
        v2_i=v_i[1]
        v3_i=v_i[2]

        if ( ((F* v2_i) + upper_bound -e2_i) > ((F*v2)+upper_bound -e2)) == True:

          #if this equation is true, there is no equilibrium
          equil=False

          #save the equation value
          pi = (F* v2_i) + upper_bound - e2_i

          #if there exists a value greater than the maximum profit value, change max
          if pi > pi_max:
            pi_max = pi
            e2_max = e2_i
            v2_max = v2_i

      except:
        print("Error")

    #reset pi and pi_max
    pi = -9999
    pi_max= -9999

    #fix e1 and e2
    for e3_i in range(lower_bound,upper_bound+1):
      try:
        #compute v_i using the new e vector
        v_i, S, r, ri, mi = cbfq(q, np.array(generate_p_values([e1,e2,e3_i],exp)), d)
        v1_i=v_i[0]
        v2_i=v_i[1]
        v3_i=v_i[2]

        if ( ((F* v3_i) + upper_bound -e3_i) > ((F*v3)+upper_bound -e3)) == True:

          #if this equation is true, there is no equilibrium
          equil=False

          #save the equation value
          pi = (F* v3_i) + upper_bound - e3_i

          #if there exists a value greater than the maximum profit value, change max
          if pi > pi_max:
            pi_max = pi
            e3_max = e3_i
            v3_max = v3_i

      except:
        print("Error")

  return [e1,e2,e3,v1,v2,v3,equil,e1_max,e2_max,e3_max,v1_max,v2_max,v3_max]

e=df.e
v=df.v
all_data=[]

for i in range(len(e)):
  data = eql_equation(e[i],v[i],F)
  all_data.append(data)

df_all_data = pd.DataFrame(all_data, columns = ['e1','e2','e3','v1','v2','v3','Equilibrium','e1_max','e2_max','e3_max','v1_max','v2_max','v3_max'])

if type_e == "symmetric_e_values":
  df_all_data.to_csv('symmetric_max_values_'+str(lower_bound)+'_'+str(upper_bound)+'.csv')
elif type_e == "asymmetric_e_values":
  df_all_data.to_csv('asymmetric_max_values_'+str(lower_bound)+'_'+str(upper_bound)+'.csv')
