f=open("first.txt","w")
d=dict()
f.write("write anything you want in here to see how many words here here are here\n")
f=open("first.txt","r")
print(f.readline())
f=open("first.txt","rt")
for line in f:
    line=line.strip()
words=line.split()
for word in words:
    if word in d:
        d[word] = d[word] + 1
    else:
        d[word] = 1
for key in list(d.keys()):
    #print(key, d[key])
f1=open("ouput.txt","w")
f1.write(key,d[key])
f1=open("output.txt","r")
print(f1.readline())

