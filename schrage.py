
import operator

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

def read_from_file(file_name):
    f = open(file_name, "r")

    line_from_file = f.readline()

    list_from_file = line_from_file.split()
    number_of_task = int(list_from_file[0])
    number_of_machine = int(list_from_file[1])

    loaded_table_from_file = []
    matrix_tasks = []

    for i in range(number_of_task):
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

    #sch.matrix_tasks[0].prep_time
    z=0
    #x,y= sch.schrange_alg(read_from_file("dane.txt"))
    x,y,z = sch.schrange_alg_parallel(read_from_file("dane.txt"))
    print(x,y,z)
    #print(matrix[2].index(max(matrix,key=operator.itemgetter(2))[2]))
