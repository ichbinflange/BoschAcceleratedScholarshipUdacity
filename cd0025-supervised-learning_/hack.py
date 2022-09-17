import numpy as np
import math
arr = np.asarray([[112, 42, 83, 119], [56, 125, 56, 49], [15, 78, 101, 43], [62, 98, 114, 108]])

print(arr.shape)

sum = 0
highest= 0
lowest=0
index=0
indexl =0
min=0
 
    
 
# finding the column sum
for i in range(arr.shape[0]) :
    for j in range(arr.shape[1]) :

        # Add the element
        sum += arr[j][i]

    # Print the column sum
    print("Sum of the column",i,"=",sum)
   
    if sum>highest:
        highest=sum
        index= i
        if lowest == 0:
            lowest=sum
        elif lowest > sum:
            lowest = sum
            indexl = i

  
    # Reset the sum
    sum = 0


arr[:,index]= np.flip(arr[:,index])
print(arr)
lowestrow = arr[:,indexl]
highestrow= arr[:,index]
print(arr)
n = math.sqrt(arr.shape[0])

n=int(n)
result=0
for i in (arr[:n,:n],arr[:n,:n]):
    print(i)
    print(np.sum(np.concatenate(i)))

print(result)