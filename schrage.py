
import operator
import numpy as np
import random
import time as t


class data:
    def __init__(self,prepare, make, delivery, task):
        self.prep_time = prepare
        self.make_time = make
        self.deliv_time = delivery
        self.task_num = task


class schrage:
    def __init__(self):
        self.tasks_number=0
        self.stages_number=3
        self.matrix_tasks=[]
        self.part_perm = []
        self.max_end_time = 0
        ######################
        self.part_perm_1 = []
        self.part_perm_2 = []
        self.max_end_time_1 = 0
        self.max_end_time_2 = 0

    def schrange_alg(self,matrix_rpq):
        self.matrix_tasks = matrix_rpq
        N_g = [] # zbior zadan do uszeregowania
        N_n = self.matrix_tasks.copy()  #zbior zadan nieuszeregowanych
        #current_time = min(N_n)[0] #najmniejszy czas przygotowania w zadaniach
        current_time =  min(N_n,key= lambda data:data.prep_time).prep_time

        while(len(N_g) != 0 or len(N_n) != 0):
            while(len(N_n) != 0 and min(N_n,key= lambda data:data.prep_time).prep_time <= current_time):#budowanie zbioru zadan gotowych do uszeregowania
                j = N_n.index(min(N_n, key= lambda data:data.prep_time))
                N_g.append(N_n.pop(j))#wrzucamy na zbior po kryterium najmniejszego czasu
            if len(N_g) == 0: #aktualizacja chwili
                #current_time = min(N_n, key= lambda data:data.prep_time).prep_time
                pass
            else:
                j = N_g.index(max(N_g, key= lambda data:data.deliv_time))#szukamy maksymalnego czasu delivery
                tmp = N_g.pop(j)#zdejmujemy to zadanie z wstepnie uszeregowanej tablicy
                #w tym miejscu nalezy dodac rownoleglosc w postaci dwoch list uszeregowan
                #zaczac od wrzucania bez kryterium, poprostu jeden tu jeden tu
                #i policzyc cmax
                #potem dopiero pobawic sie z kryterium i to bedzie tyle na wielki rownolegly rpq?
                self.part_perm.append(tmp.task_num)#appendujemy na liste rozwiazan
                current_time += tmp.make_time
                self.max_end_time = max(self.max_end_time, current_time + tmp.deliv_time)

        return self.max_end_time, self.part_perm

    def schrange_alg_parallel(self,matrix_rpq):
        self.matrix_tasks = matrix_rpq
        N_n = self.matrix_tasks.copy()

        N_g_1 = [] #zbior zadan do uszeregowania maszyna 1
        N_g_2 = [] #zbior zadan do uszeregowania maszyna 2

        current_time_1 =  min(N_n,key= lambda data:data.prep_time).prep_time #zadanie o najkrotszym czasie przygotowania
        j = N_n.index(min(N_n, key=lambda data: data.prep_time))
        temp_min_elem = N_n.pop(j) #sciagamy
        current_time_2 = min(N_n, key=lambda data: data.prep_time).prep_time #i z tego co zostalo wybieramy kolejne najkrotsze
        N_n.append(temp_min_elem) #appendujemy z powrotem


        while(len(N_g_1) != 0 or len(N_g_2) != 0 or len(N_n) != 0):
            while(len(N_n) != 0 and ((min(N_n,key= lambda data:data.prep_time).prep_time <= current_time_1) or
                                      min(N_n,key= lambda data:data.prep_time).prep_time <= current_time_2)):

                if min(N_n,key= lambda data:data.prep_time).prep_time <= current_time_1:
                    j = N_n.index(min(N_n, key=lambda data: data.prep_time))
                    N_g_1.append(N_n.pop(j))  # wrzucamy na zbior po kryterium najmniejszego czasu

                if len(N_n) != 0 and min(N_n,key= lambda data:data.prep_time).prep_time <= current_time_2:
                    j = N_n.index(min(N_n, key=lambda data: data.prep_time))
                    N_g_2.append(N_n.pop(j)) #gdybysmy chcieli wrzucic to samo zadanie, ktore przeszlo pierwszego while, to niemozliwe bo juz zdjelismy to z tablicy\

            if len(N_g_1) == 0:
                current_time_1 = min(N_n, key=lambda data: data.prep_time).prep_time
            else:
                j = N_g_1.index(max(N_g_1, key= lambda data:data.deliv_time))#szukamy maksymalnego czasu delivery
                tmp = N_g_1.pop(j)#zdejmujemy to zadanie z wstepnie uszeregowanej tablicy
                self.part_perm_1.append(tmp.task_num)#appendujemy na liste rozwiazan
                current_time_1 += tmp.make_time
                self.max_end_time_1 = max(self.max_end_time_1, current_time_1 + tmp.deliv_time)

            if len(N_g_2) == 0 and len(N_n) !=0 :
                current_time_2 = min(N_n, key=lambda data: data.prep_time).prep_time
            elif len(N_g_2) !=0 :
                j = N_g_2.index(max(N_g_2, key= lambda data:data.deliv_time))
                tmp = N_g_2.pop(j)
                self.part_perm_2.append(tmp.task_num)
                current_time_2 += tmp.make_time
                self.max_end_time_2 = max(self.max_end_time_2, current_time_2 + tmp.deliv_time)
        self.max_end_time = max(self.max_end_time_1, self.max_end_time_2)

        return self.max_end_time, self.part_perm_1, self.part_perm_2


class genetyk:

    def __init__(self):
        self.tasks_number = 0
        self.numOfGenerations = 2000
        self.numOfPopulation = 32
        self.numberOfParents = 16
        ######################
        self.bestScore = 0
        self.bestSol = None

    def solve(self, matrix):

        self.tasks_number = len(matrix)
        tasks = matrix.copy()
        self.bestScore = 10e9

        solutions = np.empty(shape=(self.numOfPopulation,self.tasks_number),dtype=data)
        print(tasks[49].task_num,tasks[49].prep_time,tasks[49].make_time,tasks[49].deliv_time)
        # init - get initial population
        for i in range(0, self.numOfPopulation):
            random.shuffle(tasks)
            solutions[i] = tasks

        for i in range(0,self.numOfGenerations):

            if i%5==0:
                print("Generation: ", i,"   Score: ",self.bestScore)
            try:
                #for k in range(0,self.tasks_number):
                #    print(self.bestSol[k].task_num)
                pass
            except:
                pass

            #asses fitness of population
            fit = self.calcTime(solutions)

            # get and remember best solution
            self.getBest(solutions,fit)

            #decide on what are the best solutions for reproduction
            matingPool = self.chooseMatingPool(solutions,fit)

            #produce offspring
            children = self.combination(matingPool)

            #random mutation
            childrenMutated = self.mutation(children)

            solutions = childrenMutated

    def mutation(self,solutions):
        mutated = np.empty(shape=(self.numOfPopulation, self.tasks_number), dtype=data)

        for i in range(0,self.numOfPopulation):
            randomSwap1 = np.random.randint(0, self.tasks_number, 1)
            randomSwap2 = np.random.randint(0, self.tasks_number, 1)

            temp = solutions[i][randomSwap1]
            solutions[i][randomSwap1] = solutions[i][randomSwap2]
            solutions[i][randomSwap2] = temp

            randomSwap1 = np.random.randint(0, self.tasks_number, 1)
            randomSwap2 = np.random.randint(0, self.tasks_number, 1)

            temp = solutions[i][randomSwap1]
            solutions[i][randomSwap1] = solutions[i][randomSwap2]
            solutions[i][randomSwap2] = temp

            mutated[i] = solutions[i]
        return mutated

    def combination(self,matingPool):
        children = np.empty(shape=(self.numOfPopulation,self.tasks_number),dtype=data)

        for i in range(0,self.numOfPopulation):
            partner1Idx = partner2Idx = 0
            randomSplitOfGenome = np.random.randint(0,self.tasks_number,1)

            while partner1Idx == partner2Idx:
                partner1Idx = np.random.randint(0, self.numberOfParents, 1)
                partner2Idx = np.random.randint(0, self.numberOfParents, 1)

            partner1 = matingPool[partner1Idx][0]
            partner2 = matingPool[partner2Idx][0]
            randomSplitOfGenome = randomSplitOfGenome[0]

            children[i] = partner2
            #for j in range(0,randomSplitOfGenome):
            #    children[i,j] = partner1[j]

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

                #print(i,time,sol[j, i].task_num, sol[j, i].prep_time, sol[j,i].make_time, sol[j,i].deliv_time)
                #t.sleep(0.1)

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
                #print(M1EndOfTask,M2EndOfTask)
                time =          np.max([time,np.min([M1EndOfTask, M2EndOfTask])]) #next time jump when one of the machines finishes task

                timeLonger =    np.max([M1EndOfDelivery, M2EndOfDelivery])

                timeLastDelivery[j] = np.max([timeLastDelivery[j],timeLonger])  # to time t, add time of delivery, check if this will be the last delivery, if yes - remember this time

        return timeLastDelivery

    def checkShortestPossible(self,tasks):
        self.tasks_number = len(tasks)
        sumOfTimes = np.zeros(shape=self.tasks_number)
        for i in range(0,self.tasks_number):
            sumOfTimes[i] = tasks[i].prep_time + tasks[i].make_time + tasks[i].deliv_time
        print(np.argmax(sumOfTimes))
        print(sumOfTimes[np.argmax(sumOfTimes)])



def read_from_file(file_name):
    f = open(file_name, "r")

    line_from_file = f.readline()

    list_from_file = line_from_file.split()
    number_of_tasks = int(list_from_file[0])
    #?  number_of_machine = int(list_from_file[1])

    loaded_table_from_file = []
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



if __name__ == "__main__":

    sch = schrage()
    gen = genetyk()
    #sch.matrix_tasks[0].prep_time
    #x,y= sch.schrange_alg(read_from_file("dane.txt"))
    #x,y,z = sch.schrange_alg_parallel(read_from_file("dane.txt"))
    #data = read_from_file("dane.txt")
    #print(x,y,z)
    #print(data[5].task_num)
    #print(matrix[2].index(max(matrix,key=operator.itemgetter(2))[2]))
    #gen.checkShortestPossible(read_from_file("dane2.txt"))

    gen.solve(read_from_file("dane.txt"))

    # dane (dane008) - najlepiej możliwie 3605 a z materiałów wynika, że optymalnie 3633
    #dane 2 - najlepiej możliwie 3026