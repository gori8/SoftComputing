#!/usr/bin/python
import sys

res = []
n = 0
with open('C:/soft/SoftComputing/res.txt') as file:	
    data = file.read()
    lines = data.split('\n')
    for id, line in enumerate(lines):
        if(id>0):
            cols = line.split(' ')
            if(cols[0] == ''):
                continue
            cols[1] = cols[1].replace('\r', '')
            res.append(float(cols[1]))
            n += 1

correct = 0
student = []
student_results = []
with open("C:/soft/SoftComputing/out.txt") as file:
    data = file.read()
    lines = data.split('\n')
    for id, line in enumerate(lines):
        cols = line.split(' ')
        if(cols[0] == ''):
            continue
        if(id==0):
            student = line  
        elif(id>1):
            cols[1] = cols[1].replace('\r', '')
            student_results.append(float(cols[1]))

diff = 0
for index, res_col in enumerate(res):
    diff += abs(res_col - student_results[index])
percentage = 100 - abs(diff/sum(res))*100

print(student)
print('Procenat tacnosti:\t'+str(round(percentage, 2)) + '%')
print('Ukupno:\t'+str(n))