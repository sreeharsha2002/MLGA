import numpy as np
import random
import json
from client import get_errors, submit
forSubmission = 1
smallNumber = 10**(-18)
counter = 0
fitnessFunctionChanged = 0

def getProbabilityArray(arr):
    return list(np.divide(arr, np.sum(arr)))


def crossover(pair, selectedParents):
    A = selectedParents[pair[0]]
    B = selectedParents[pair[1]]
    C = A.copy()
    D = B.copy()

    number_of_different_genes = 0

    for i in range(numParameters):
        if(abs(C[i] - D[i]) >= smallNumber):
            number_of_different_genes += 1

    swap_counter = 0

    while swap_counter <= (number_of_different_genes/2) and number_of_different_genes != 0:
        for i in range(numParameters):
            if (abs(C[i] - D[i]) >= smallNumber and abs(C[i] - B[i]) >= smallNumber):
                u = random.random()
                if(u <= 0.5):
                    C[i] = B[i]
                    D[i] = A[i]
                    swap_counter += 1
    return C, D


def fitnessValue(arr):
    if(forSubmission == 1):
        e = get_errors(
            'ZTtfwlacwCWs7DiHl0eP4MG9kYi8f9yvaeaE1qdRkj58Zao3Ek', list(arr))
    else:
        e = [1, 2]
    global counter
    counter += 1
    aa = 10**(-8)
    bb = 1
    # (aa*( e[0]+ e[1])) + bb*abs(e[0]-e[1])
    return (((10**15)/(  bb*abs(e[0]-e[1]) + aa*(e[0]+e[1])) ) ) , e 


overfitArray = [0.0, -0.31238334448733596, 0.0, 0.0, -1.2723354550089536, 0.0, 0.05400445338789047, 0.0, 0.0, 0.0, 0.0]

numParameters = 11
population = 15
numParentsMating = 5
parentArray = []
fitnessArray = []
try:
    with open('parentFile1.txt', 'r') as f:
        parentArray = json.load(f)
    with open('parentFitness.txt', 'r') as f:
        fitnessArray = json.load(f)
except:
    parentArray.append(np.array(overfitArray))
    ft, error = fitnessValue(overfitArray)
    print("")
    print("")
    print("")
    print("New trial")
    print(f"initial vector : {list(overfitArray)}")
    print(f"  train error = { error[0] } ")
    print(f" Validation error = { error[1] }")
    print("")
    print("")
    print(f"initial population  : \n  ")
    print(f" vector : {list(overfitArray)}  ,  ")

    print(f"  train error = { error[0] } ")
    print(f" Validation error = { error[1] }")
    print("")
    fitnessArray.append(ft)
    for i in range(population-1):
        array = np.array(overfitArray.copy())
        normarray = np.random.normal(array, abs((0.33)-array))
        ind = np.random.choice(11, 5,replace=True)
        array[ind] = normarray[ind]
        for i in range(len(array)):
            if(array[i] >=10):
                array[i] = 9
            if(array[i] <= -10):
                array[i] = -9 
        parentArray.append(array)
        ft , error = fitnessValue(array)
        fitnessArray.append(ft)
        print(f"vector : {list(array)} , ")
        print(f"  train error = { error[0] } ")
        print(f" Validation error = { error[1] }")
        print("")
    



parentArray = np.array(parentArray)
fitnessArray = np.array(fitnessArray)

indices = np.argsort((-1)*fitnessArray)
parentArray = parentArray[indices]
fitnessArray = fitnessArray[indices]
try:
    with open('top101.txt', 'r') as f:
        top10 = json.load(f)

    with open('top10Fitness.txt', 'r') as f:
        top10fitness = json.load(f)
except:
    top10 = parentArray[0:10]
    top10fitness = fitnessArray[0:10]
    
top10 = np.array(top10)
top10fitness = np.array(top10fitness)

if(fitnessFunctionChanged == 1):
    fitnessArray = [] 
    for i in parentArray:
        ft, error = fitnessValue(i)
        fitnessArray.append(ft)
    top10fitness= []
    for i in top10:
        ft, error = fitnessValue(i)
        top10fitness.append(ft)
    fitnessArray = np.array(fitnessArray)
    top10fitness = np.array(top10fitness )
    top10indices = np.argsort((-1)*top10fitness)
    top10 = top10[top10indices]
    top10fitness = top10fitness[top10indices]
    

num_generations = 5
numParentsMating = 5



for generation in range(num_generations):
    print("Generation : ", generation)
    
    print(f"Parents for the generation are: ")
    print('[')
    for iii in parentArray:
        print(f" {list(iii)}, ")
        print("")
    print(']')
    indices = np.argsort((-1)*fitnessArray)
    parentArray = parentArray[indices]
    fitnessArray = fitnessArray[indices]

    selectedParents = parentArray[0:numParentsMating]
    selectedFitness = fitnessArray[0:numParentsMating]
    ProbabilityArray = getProbabilityArray(selectedFitness)
    matingParents = []
    for i in range(numParentsMating):
        matingParents.append(np.random.choice(numParentsMating, 2, p=ProbabilityArray, replace=False))

    childrenArray = []
    # crossover
    print("Cross over")
    print("")
    for pair in matingParents:
        child1, child2 = crossover(pair, selectedParents)
        print(f"Parents : { list(selectedParents[pair[0]]) } X { list(selectedParents[pair[1]]) } ------>  ")
        print(f"Children : { list(child1) } + { list(child2) }")
        print("<------------------->")
        print("")
        childrenArray.append(child1)
        childrenArray.append(child2)
    childrenArray2 = []
    childFitness = []
    
    print("MUTATION ")
    print("")

    for i in childrenArray:
        array = np.array(i)
        normarray = np.random.uniform((-5/100)*array, (5/100)*array)
        ind = np.random.choice(11, random.randint(1, 10),replace=False)
        randind = np.random.choice(11, 3,replace=False)
        array[ind] += normarray[ind]
        for index in randind:
            if array[index]==0:
                array[index]=np.random.normal(0,0.33*(10**(-12)))
        for index in range(len(array)):
            if(array[index] >=10):
                array[index] = 9
            if(array[index] <= -10):
                array[index] = -9 

        childrenArray2.append(array)
        print(f"Mutated child : { list(array) } ")
        ft, error = fitnessValue(array)
        
        print(f"  train error = { error[0] } ")
        print(f" Validation error = { error[1] }")
        print("")
        childFitness.append(ft)
        

    childrenArray2 = np.array(childrenArray2)
    childFitness = np.array(childFitness)

    childindices = np.argsort((-1)*childFitness)
    childrenArray2 = childrenArray2[childindices]
    childFitness = childFitness[childindices]

    top10 = np.concatenate((top10, childrenArray2))
    top10fitness = np.concatenate((top10fitness, childFitness))

    top10indices = np.argsort((-1)*top10fitness)
    top10 = top10[top10indices]
    top10fitness = top10fitness[top10indices]

    top10 = top10[0:10]
    top10fitness = top10fitness[0:10]

    selectedBatch1 = np.concatenate(
        (selectedParents[0:3], childrenArray2[0:7]))
    selectedBatch1Fitness = np.concatenate(
        (selectedFitness[0:3], childFitness[0:7]))

    selectedBatch2 = np.concatenate((parentArray[6:15], childrenArray2[8:10]))
    selectedBatch2Fitness = np.concatenate(
        (fitnessArray[6:15], childFitness[8:10]))

    sortindices = np.argsort((-1)*selectedBatch2Fitness)
    selectedBatch2 = selectedBatch2[sortindices]
    selectedBatch2Fitness = selectedBatch2Fitness[sortindices]

    selectedBatch2 = selectedBatch2[0:5]
    selectedBatch2Fitness = selectedBatch2Fitness[0:5]

    finalBatch = np.concatenate((selectedBatch1, selectedBatch2))
    finalBatchFitness = np.concatenate(
        (selectedBatch1Fitness, selectedBatch2Fitness))
    parentArray = np.array(finalBatch)
    fitnessArray = np.array(finalBatchFitness)

if(forSubmission == 1):
    
    with open('parentFile1.txt', 'w') as f:
        json.dump(parentArray.tolist(), f)
    with open('top101.txt', 'w') as f:
        json.dump(top10.tolist(), f)
    with open('parentFitness.txt', 'w') as f:
        json.dump(fitnessArray.tolist(), f)
    with open('top10Fitness.txt', 'w') as f:
        json.dump(top10fitness.tolist(), f)    
    ee = get_errors(
        'ZTtfwlacwCWs7DiHl0eP4MG9kYi8f9yvaeaE1qdRkj58Zao3Ek', list(np.average(top10,axis=0)))
    print("Happening")
else:
    ee = [1, 2]
if(forSubmission == 1):
    submit('ZTtfwlacwCWs7DiHl0eP4MG9kYi8f9yvaeaE1qdRkj58Zao3Ek', list(np.average(top10,axis=0)))
    print(f"Top10 are: ")
    print('[')
    for iii in top10:
        print(f" {list(iii)}, ")
        ft,error=fitnessValue(iii)
        print(f"  train error = { error[0] } ")
        print(f" Validation error = { error[1] }")
        print("")
    print(']')
    print("Average of top10 best:")
    print(list(np.average(top10,axis=0)))
    print("Errors")
    print((ee[0]/(10**10)) , (ee[1]/(10**10)))
    print(abs(ee[0]-ee[1]))
    print(f"Counter:{counter}")
print("Program Finished")