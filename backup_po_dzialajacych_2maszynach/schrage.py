
import operator
import numpy as np
import random
import time as t
import matplotlib.pyplot as plt
from statistics import mean

class data:
    def __init__(self,prepare, make, delivery, task):
        self.prep_time = prepare
        self.make_time = make
        self.deliv_time = delivery
        self.task_num = task


class schrage:

    def __init__(self):
        self.tasks_number = 0
        self.stages_number = 3
        self.matrix_tasks = []
        self.part_perm = []
        self.max_end_time = 0
        self.max_end_time_parallel = 0
        ######################
        self.part_perm_1 = []
        self.part_perm_2 = []
        self.max_end_time_1 = 0
        self.max_end_time_2 = 0
        ######################
        self.timeOfExecution = 0
        self.bestSol = None
        self.bestPossibleTime = 0

    def schrange_alg(self,matrix_rpq):

        timeStart = t.time()
        self.matrix_tasks = matrix_rpq
        N_g = []                        # zbior zadan do uszeregowania
        N_n = self.matrix_tasks.copy()  #zbior zadan nieuszeregowanych

        current_time =  min(N_n,key=lambda data:data.prep_time).prep_time

        while len(N_g) != 0 or len(N_n) != 0:
            while len(N_n) != 0 and min(N_n,key=lambda data:data.prep_time).prep_time <= current_time:      #budowanie zbioru zadan gotowych do uszeregowania
                j = N_n.index(min(N_n, key=lambda data:data.prep_time))
                N_g.append(N_n.pop(j))                                                                      #wrzucamy na zbior po kryterium najmniejszego czasu
            if len(N_g) == 0:                                                                               #aktualizacja chwili
                current_time = min(N_n, key=lambda data:data.prep_time).prep_time
                pass
            else:
                j = N_g.index(max(N_g, key=lambda data:data.deliv_time))                                    #szukamy maksymalnego czasu delivery
                tmp = N_g.pop(j)                                                                            #zdejmujemy to zadanie z wstepnie uszeregowanej tablicy
                self.part_perm.append(tmp.task_num)                                                         #appendujemy na liste rozwiazan
                current_time += tmp.make_time
                self.max_end_time = max(self.max_end_time, current_time + tmp.deliv_time)

        self.bestSol = self.composeSolutionFromPermutation(self.part_perm, matrix_rpq)
        self.max_end_time_parallel = self.calcTimeParallel(self.bestSol)

        timeStop = t.time()
        self.timeOfExecution = timeStop - timeStart


    def checkShortestPossible(self, tasks):
        self.tasks_number = len(tasks)
        sumOfTimes = np.zeros(shape=self.tasks_number)
        for i in range(0, self.tasks_number):
            sumOfTimes[i] = tasks[i].prep_time + tasks[i].make_time + tasks[i].deliv_time
        # print(np.argmax(sumOfTimes))
        #print("Shortest possible: ", sumOfTimes[np.argmax(sumOfTimes)])
        self.bestPossibleTime = sumOfTimes[np.argmax(sumOfTimes)]

    def composeSolutionFromPermutation(self,permutation,tasks):
        solution = []
        for i in range(0,len(permutation)):
            for j in range(0,len(tasks)):
                if tasks[j].task_num == permutation[i]:
                    solution.append(tasks[j])
        return solution

    def calcTimeParallel(self,sol):

        timeLastDelivery = 0
        M1EndOfTask = 0
        M2EndOfTask = 0
        M1EndOfDelivery = 0
        M2EndOfDelivery = 0
        time = 0

        for i in range(0,len(sol)):
            time = np.max([time,sol[i].prep_time])  #we are in time t, if prep time Pt of current task is Pt > t, then jump with time to Pt
            M1Ready = True if M1EndOfTask <= time else False
            M2Ready = True if M2EndOfTask <= time else False
            if M1Ready:
                M1EndOfTask = time + sol[i].make_time
                M1EndOfDelivery = np.max([M1EndOfDelivery,M1EndOfTask + sol[i].deliv_time])
            elif M2Ready:
                M2EndOfTask = time + sol[i].make_time
                M2EndOfDelivery = np.max([M2EndOfDelivery, M2EndOfTask + sol[i].deliv_time])
            else:
                print("exception occurred")
            time =          np.max([time,np.min([M1EndOfTask, M2EndOfTask])]) #next time jump when one of the machines finishes task
            timeLonger =    np.max([M1EndOfDelivery, M2EndOfDelivery])
            timeLastDelivery = np.max([timeLastDelivery,timeLonger])  # to time t, add time of delivery, check if this will be the last delivery, if yes - remember this time

        return timeLastDelivery

    def printRaport(self):
        print("Solution found: ", str(self.part_perm))
        print("Best possible time: ", str(self.bestPossibleTime))
        print("C_max: ", str(self.max_end_time))
        print("C_max_parallel: ", str(self.max_end_time_parallel))
        print("Time of execution: ", str(self.timeOfExecution), "s")

class genetyk:

    def __init__(self,numOfMachines):
        self.tasks_number = 0
        self.numOfPopulation = 16
        self.numberOfParents = 4
        self.numOfMachines = numOfMachines
        ######################
        self.bestScore = 10e9
        self.bestSol = None
        self.bestPossibleScore = 0
        ######################
        self.historyOfOptimisation = []
        self.timeOfExecution = 0.0
        self.iterations = 0
        self.iterationsMax = 8000
        self.iterationsWithNoChangeMax = 600
        self.iterationsWithNoChange = 0

    def clearData(self):
        self.bestScore = 10e9
        self.bestSol = None
        ######################
        self.historyOfOptimisation = []
        self.iterations = 0
        self.iterationsWithNoChange = 0

    def solve(self, matrix):

        self.clearData()
        timeStart = t.time()
        tasks = matrix.copy()
        self.tasks_number = len(matrix)

        solutions = np.empty(shape=(self.numOfPopulation,self.tasks_number),dtype=data)

        # init - get initial population
        for i in range(0, self.numOfPopulation):
            random.shuffle(tasks)
            solutions[i] = tasks

        self.iterations = 0
        while self.bestPossibleScore < self.bestScore and self.iterations < self.iterationsMax and self.iterationsWithNoChange < self.iterationsWithNoChangeMax:

            self.iterations += 1
            self.iterationsWithNoChange += 1

            if self.iterations >= 2:
                self.historyOfOptimisation.append(self.bestScore)
                #self.printBestSol()

            if self.iterations % 10 == 0:
                print("Generation: ", self.iterations,"   Score: ",self.bestScore)

            #asses fitness of population
            fit = self.calcTime(solutions)

            # get and remember best solution
            self.getBest(solutions,fit)

            #decide on what are the best solutions for reproduction
            matingPool = self.chooseMatingPool(solutions,fit)

            #produce offspring
            children = self.copyParents(matingPool)

            #random mutation
            childrenMutated = self.mutation(children)

            solutions = childrenMutated

        timeStop = t.time()
        self.timeOfExecution = timeStop - timeStart

        print("\n")
        if not self.bestPossibleScore < self.bestScore:
            print("Best possible score found")
        if not self.iterations < self.iterationsMax:
            print("Max num of generations reached")
        if not self.iterationsWithNoChange < self.iterationsWithNoChangeMax:
            print("Max num of generations without a change of score reached")

    def mutation(self,solutions):
        mutated = np.empty(shape=(self.numOfPopulation, self.tasks_number), dtype=data)

        for i in range(0,self.numOfPopulation):
            randomSwap1 = np.random.randint(0, self.tasks_number, 1)
            randomSwap2 = np.random.randint(0, self.tasks_number, 1)

            temp = solutions[i][randomSwap1]
            solutions[i][randomSwap1] = solutions[i][randomSwap2]
            solutions[i][randomSwap2] = temp

            mutated[i] = solutions[i]
        return mutated

    def copyParents(self,matingPool):

        children = np.empty(shape=(self.numOfPopulation, self.tasks_number), dtype=data)
        for i in range(0, self.numOfPopulation):

            partner1Id = np.random.randint(0, self.numberOfParents, 1)
            partner1 = matingPool[partner1Id][0]
            children[i] = partner1
        return children

    def combination(self,matingPool):
        children = np.empty(shape=(self.numOfPopulation,self.tasks_number),dtype=data)

        for i in range(0,self.numOfPopulation):
            partner1Id = partner2Id = 0

            while partner1Id == partner2Id:
                partner1Id = np.random.randint(0, self.numberOfParents, 1)
                partner2Id = np.random.randint(0, self.numberOfParents, 1)

            partner1 = matingPool[partner1Id][0]
            partner2 = matingPool[partner2Id][0]

            tasksToMove = []
            for j in range(0,self.tasks_number):
                if partner1[j].task_num ==  partner2[j].task_num:
                    children[i, j] = partner1[j]
                else:
                    tasksToMove.append(partner1[j])

            random.shuffle(tasksToMove)

            for k in range(0,self.tasks_number):
                if children[i, k] == None:
                    children[i, k] = tasksToMove[0]
                    tasksToMove.pop(0)

        return children

    def chooseMatingPool(self,solutons,fiteness):

        matingGroup = np.empty(shape=(self.numberOfParents,len(solutons[0])),dtype=data)
        score = fiteness.copy()

        for i in range(0,self.numberOfParents):
            bestIdx = np.argmin(score)
            matingGroup[i] = solutons[bestIdx]
            score[bestIdx] = 10e9
        return matingGroup

    def getBest(self,solutions,fit):
        bestIdx = np.argmin(fit)
        if fit[bestIdx] < self.bestScore:
            self.iterationsWithNoChange = 0
            self.bestScore = fit[bestIdx]
            self.bestSol = solutions[bestIdx]

    def calcTime(self,sol):
        timeLastDelivery = np.zeros(shape=len(sol))

        for j in range(0,len(sol)):
            M1EndOfTask = 0
            M2EndOfTask = 0
            M1EndOfDelivery = 0
            M2EndOfDelivery = 0
            time = 0

            for i in range(0,len(sol[0])):
                time = np.max([time,sol[j,i].prep_time])  #we are in time t, if prep time Pt of current task is Pt > t, then jump with time to Pt

                M1Ready = True if M1EndOfTask <= time else False
                M2Ready = True if M2EndOfTask <= time else False

                if M1Ready:
                    M1EndOfTask = time + sol[j,i].make_time
                    M1EndOfDelivery = np.max([M1EndOfDelivery,M1EndOfTask + sol[j,i].deliv_time])
                elif M2Ready:
                    M2EndOfTask = time + sol[j,i].make_time
                    M2EndOfDelivery = np.max([M2EndOfDelivery, M2EndOfTask + sol[j, i].deliv_time])
                else:
                    print("exception occurred")

                time =          np.max([time,np.min([M1EndOfTask, M2EndOfTask])]) #next time jump when one of the machines finishes task

                timeLonger =    np.max([M1EndOfDelivery, M2EndOfDelivery])

                timeLastDelivery[j] = np.max([timeLastDelivery[j],timeLonger])  # to time t, add time of delivery, check if this will be the last delivery, if yes - remember this time

        return timeLastDelivery

    def checkShortestPossible(self,tasks):
        self.tasks_number = len(tasks)
        sumOfTimes = np.zeros(shape=self.tasks_number)
        for i in range(0,self.tasks_number):
            sumOfTimes[i] = tasks[i].prep_time + tasks[i].make_time + tasks[i].deliv_time
        #print(np.argmax(sumOfTimes))
        print("Shortest possible: ", sumOfTimes[np.argmax(sumOfTimes)])
        self.bestPossibleScore = sumOfTimes[np.argmax(sumOfTimes)]

    def plotHistory(self):
        plt.plot(self.historyOfOptimisation)
        plt.title('Optymalizacja algorytmem genetycznym, n= ' + str(self.tasks_number) + ', t= ' + str(int(self.timeOfExecution)) + 's')
        plt.ylabel('Cmax')
        plt.xlabel('Generacja')
        plt.show()

    def optimalTimeOverride(self,setVal):
        self.bestPossibleScore = setVal

    def printBestSol(self):
        bestSol = []
        for i in range(0, self.tasks_number):
            bestSol.append(self.bestSol[i].task_num)
        print("Found solution: ", bestSol)
        print("Time of execution: ", self.timeOfExecution, "s")

    def checkBestSol(self):
        time = self.calcTime(np.array([self.bestSol]))
        print('Best time possible is: ',self.bestPossibleScore)
        print('Best time when optimising: ', self.bestScore)
        print('Time of output solution: ', time[0])


def read_from_file(file_name):
    f = open(file_name, "r")

    line_from_file = f.readline()

    list_from_file = line_from_file.split()
    number_of_tasks = int(list_from_file[0])

    matrix_tasks = []

    for i in range(number_of_tasks):
        line_from_file = f.readline()
        list_from_file = line_from_file.split()

        prep_time = int(list_from_file[0])
        make_time = int(list_from_file[1])
        deliv_time = int(list_from_file[2])
        rpq = data(prep_time, make_time, deliv_time, i)
        matrix_tasks.append(rpq)

    return matrix_tasks


def plotResults_1():
    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))

    # set height of bar
    Schrage = [3026, 3652, 3309, 3172, 3618, 3435, 3821, 3605]
    Gen = [3026, 3643, 3309, 3172, 3618, 3413, 3798, 3605]

    # Set position of bar on X axis
    br1 = np.arange(len(Schrage))
    br2 = [x + barWidth for x in br1]

    max_y_lim = 4000
    min_y_lim = 3000
    plt.ylim(min_y_lim, max_y_lim)

    # Make the plot
    plt.bar(br1, Schrage, color='#89ABE3FF', width=barWidth,
            edgecolor='grey', label='Schrage')
    plt.bar(br2, Gen, color=(1, 0.74, 0.8), width=barWidth,
            edgecolor='grey', label='Teoretycznie możliwy najlepszy wynik')

    # Adding Xticks
    plt.xlabel('Zestawy Danych', fontweight='bold', fontsize=15)
    plt.ylabel('Cmax odnalezionego rozwiązania', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(Schrage))],
               ['001', '002', '003', '004', '005','006', '007', '008'])
    plt.title("Wyniki optymalizacji ")

    plt.legend()
    plt.show()


def plotResults_2():
    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))

    # set height of bar
    baseline = [3643, 3643, 3798]
    Schrage = [3652, 5985, 8493]
    Gen = [3643, 5178, 7641]

    # Set position of bar on X axis
    br1 = np.arange(len(Schrage))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    max_y_lim = 9000
    min_y_lim = 3000
    plt.ylim(min_y_lim, max_y_lim)

    # Make the plot
    plt.bar(br1, baseline, color='#D8E389', width=barWidth,
            edgecolor='grey', label='Teoretycznie możliwy najlepszy wynik')
    plt.bar(br3, Schrage, color='#89ABE3FF', width=barWidth,
            edgecolor='grey', label='Algorytm Schrage')
    plt.bar(br2, Gen, color=(1, 0.74, 0.8), width=barWidth,
            edgecolor='grey', label='Algorytm Genetyczny')

    # Adding Xticks
    plt.xlabel('Zestawy Danych', fontweight='bold', fontsize=15)
    plt.ylabel('Cmax odnalezionego rozwiązania', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(Schrage))],
               ['010', '011', '012'])
    plt.title("Wyniki optymalizacji ")

    plt.legend()
    plt.show()


def calcDiff1():
    Schrage = [3026, 3652, 3309, 3172, 3618, 3435, 3821, 3605]
    Gen = [3026, 3643, 3309, 3172, 3618, 3413, 3798, 3605]
    res = []
    for i in range(0,len(Gen)):
        res.append(((Schrage[i] / Gen[i])-1.0)*100)
    print(res)


def calcDiff2():
    baseline = [3643,3643,3798]
    Schrage = [3652, 5985, 8493]
    Gen = [3643, 5178, 7641]

    res = []
    for i in range(0,len(Gen)):
        res.append(((Schrage[i] / baseline[i])-1.0)*100)
    print(res)


if __name__ == "__main__":


    #plotResults_2()
    #calcDiff2()

    executeSchrage = True
    executeGenetic = True
    testData = read_from_file("data011.txt")

    if executeSchrage:
        sch = schrage()
        sch.checkShortestPossible(testData)
        sch.schrange_alg(testData)
        sch.printRaport()
        print("\n")

    if executeGenetic:
        gen = genetyk(4)
        gen.checkShortestPossible(testData)
        gen.solve(testData)
        gen.checkBestSol()
        gen.printBestSol()
        gen.plotHistory()

