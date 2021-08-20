import math
import random
import string
from numba import jit, cuda 
import time
from timeit import default_timer as timer 
import numpy as np

POPULATION_SIZE = 20                                                            #Parametr mowiacy o wielkosci populacji
PARAM_NUMBER_OF_POSSIBLES = 20                                                  #Liczba kombinacji pojedynczego parametru
PARAM1_JUMP = 5                                                                 #Okresla zakres i rozdzielczosc parametru 1
PARAM2_JUMP = 5                                                                 #Okresla zakres i rozdzielczosc parametru 2
NUMBER_OF_POPULATION_TO_CROSS = 3                                               #Okresla liczbe osobnikow wykorzystana do procesu krzyzowania
MAKE_MUTATION = 4                                                               #Z jakim prawdopodobienstwem ma wystepowac mutacja 0-100% 10-10% 100-1%
x = 0
y = 1
z = 2


class Genetic:
    def __init__(self, axis1, axis2, axis3, axis4, axis5, axis6, big_step_resolution, short_step_resolution, objec_pos, end_pos, enemy1_pos, enemy2_pos):
        # number of input, hidden, and output nodes
        self.axis1 = axis1 
        self.axis2 = axis2 
        self.axis3 = axis3 
        self.axis4 = axis4 
        self.axis5 = axis5 
        self.axis6 = axis6 
        #self.object_to_move = object_to_move
        self.big_step_resolution = big_step_resolution
        self.short_step_resolution = short_step_resolution
        self.objec_pos = objec_pos
        self.end_pos = end_pos
        self.enemy1_pos = enemy1_pos
        self.enemy2_pos = enemy2_pos
        self.population = []
        self.bestIndividual = 0
        
    def GenerateFirstPopulation(self):
        list_of_population = []
        for p1 in range(1, PARAM_NUMBER_OF_POSSIBLES+1, 1):
            #for p2 in range(1 , PARAM_NUMBER_OF_POSSIBLES+1, 1):
            list_of_population.append([random.uniform(0.05, random.uniform(0.05, 0.95)), random.uniform(0.01, random.uniform(0.01, 0.1))])
                
        self.population = list_of_population
        return list_of_population
                
    def GetBestIndividual(self, list_of_population_with_score):
        last_score = 1
        start_array = POPULATION_SIZE
        end_array = POPULATION_SIZE*2
        element_to_remove = None
        list_of_best_score = list_of_population_with_score
        for i in range(POPULATION_SIZE-NUMBER_OF_POPULATION_TO_CROSS):
            last_score = 1
            for index in range(start_array, end_array, 1):
                if list_of_best_score[index] > last_score:
                    last_score = list_of_best_score[index]
                    element_to_remove = index
            list_of_best_score.pop(element_to_remove)
            list_of_best_score.pop(element_to_remove-start_array)
            start_array = start_array - 1
            end_array = end_array - 2
        for j in range(NUMBER_OF_POPULATION_TO_CROSS):
            list_of_best_score.pop(NUMBER_OF_POPULATION_TO_CROSS)
            
          
        return list_of_best_score
    
    def CrossingBestIndividual(self, list_of_population_to_cross):
        my_array_of_population = np.array(list_of_population_to_cross)
        new_population = []
        for cross in range(POPULATION_SIZE):
            first_individual = random.randint(0, NUMBER_OF_POPULATION_TO_CROSS-1)
            second_individual = random.randint(0, NUMBER_OF_POPULATION_TO_CROSS-1)
            while first_individual == second_individual:
                second_individual = random.randint(0, NUMBER_OF_POPULATION_TO_CROSS-1)
            cross_param_option = random.randint(0, 1)
            if cross_param_option == 1:
                new_population.append([my_array_of_population[first_individual][0], my_array_of_population[second_individual][1]])
            elif cross_param_option == 0:
                new_population.append([my_array_of_population[second_individual][0], my_array_of_population[first_individual][1]])
            
        return new_population
    
    def Mutation(self, lis_of_population_after_crossing):
        my_array_of_population = np.array(lis_of_population_after_crossing)
        new_population = []
        for mutation in range(POPULATION_SIZE):
            mutation_param_option = random.randint(0, 3)
            make_mutation = random.randint(0, MAKE_MUTATION)
            MUTATION_SIZE_PARAM1 = random.uniform(0, random.uniform(0, 0.1))
            MUTATION_SIZE_PARAM2 = random.uniform(0, random.uniform(0, 0.015))
            if make_mutation == 0:
                if mutation_param_option == 0:
                    new_population.append([(my_array_of_population[mutation][0])-MUTATION_SIZE_PARAM1, my_array_of_population[mutation][1]])
                elif mutation_param_option == 1:
                    new_population.append([(my_array_of_population[mutation][0])+MUTATION_SIZE_PARAM1, my_array_of_population[mutation][1]])
                elif mutation_param_option == 2:
                    new_population.append([my_array_of_population[mutation][0], (my_array_of_population[mutation][1])-MUTATION_SIZE_PARAM2])
                elif mutation_param_option == 3:
                    new_population.append([my_array_of_population[mutation][0], (my_array_of_population[mutation][1])+MUTATION_SIZE_PARAM2])
            else:
                new_population.append([my_array_of_population[mutation][0], my_array_of_population[mutation][1]])
                    
        self.population = new_population
        return new_population
    
    def SetBestIndividual(self, bestIndividualToSave):
        self.bestIndividual = bestIndividualToSave
        
    def SaveDataToTxt(self):
        next_line="\n"
        file = open("MyDataToTeaching.txt", "a")
        file.writelines(str(self.axis1))
        file.writelines(next_line)
        file.writelines(str(self.axis2))
        file.writelines(next_line)
        file.writelines(str(self.axis3))
        file.writelines(next_line)
        file.writelines(str(self.axis4))
        file.writelines(next_line)
        file.writelines(str(self.axis5))
        file.writelines(next_line)
        file.writelines(str(self.axis6))
        file.writelines(next_line)
        file.writelines(str(self.big_step_resolution))
        file.writelines(next_line)
        file.writelines(str(self.short_step_resolution))
        file.writelines(next_line)
        file.writelines(str(self.objec_pos[x]))
        file.writelines(next_line)
        file.writelines(str(self.objec_pos[y]))
        file.writelines(next_line)
        file.writelines(str(self.objec_pos[z]))
        file.writelines(next_line)
        file.writelines(str(self.end_pos[x]))
        file.writelines(next_line)
        file.writelines(str(self.end_pos[y]))
        file.writelines(next_line)
        file.writelines(str(self.end_pos[z]))
        file.writelines(next_line)
        file.writelines(str(self.enemy1_pos[x]))
        file.writelines(next_line)
        file.writelines(str(self.enemy1_pos[y]))
        file.writelines(next_line)
        file.writelines(str(self.enemy1_pos[z]))
        file.writelines(next_line)
        file.writelines(str(self.enemy2_pos[x]))
        file.writelines(next_line)
        file.writelines(str(self.enemy2_pos[y]))
        file.writelines(next_line)
        file.writelines(str(self.enemy2_pos[z]))
        file.writelines(next_line)
        file.writelines(str(self.bestIndividual[0]))
        file.writelines(next_line)
        file.writelines(str(self.bestIndividual[1]))
        file.writelines(next_line)
        file.close()
            

myGenetic = Genetic(1, 1, 1, 1, 1, 1, 0.01, 0.003, [0.8, 0.2, 0.4], [0.8, 0.2, 0.4], [0.8, 0.2, 0.4], [0.8, 0.2, 0.4])
if __name__ == '__main__':
    data = myGenetic.GenerateFirstPopulation()
    print("Population: " + str(myGenetic.population))
    for i in range(100):
        data.append(random.randint(100, 1200))
    print("Generate: "+str(data))
    best = myGenetic.GetBestIndividual(data)
    print("Best: "+str(best))
    cross = myGenetic.CrossingBestIndividual(best)
    print("Cross: "+str(cross))
    mutation = myGenetic.Mutation(cross)
    myGenetic.SetBestIndividual(best[1])
    print("Mutation: " + str(mutation))
    print("Population1: " + str(myGenetic.population))
    print("Data: " + str(myGenetic.axis1) + " " + str(myGenetic.bestIndividual))
    myGenetic.SaveDataToTxt()
