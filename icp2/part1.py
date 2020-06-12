a=[]
b=[]
n=int(input("enter number of students"))
for i in range(0,int(n)):
    k=float(input("enter value in lbs"))
    a.append(k)
print(a)
for i in range(0,int(n)):
    c=(float(0.4534)*a[i])
    b.append(c)
print(b)
