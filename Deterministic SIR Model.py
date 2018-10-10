# -*- coding: utf-8 -*-
"""
@authors: Kaixin and Suzan

deterministc SIR model

"""
import numpy as np
import matplotlib.pyplot as plt
import probability 
from matplotlib import mpl

def main():
    """
    Simulate the spread of disease n times, for a time period of population.time. 
    """
    # run through the  different combinations of probabilities
    probSus_vector=[0.1,0.5,0.9]
    probInf_vector=[0.1,0.5,0.9]
    for k in range(0,3):
        for m in range(0,3):
            array_S_total = []
            array_I_total = []
            array_R_total = []
            
            # simulate a hundred times
            rep = 1              
            while rep <= 100:
                population = Population() 
                population.probSusceptible=probSus_vector[k]
                population.probInfected=probInf_vector[m]  
                population.initial_values()
                # simulate the spread of disease over time
                t = 0
                while t <= population.time:
                     population.show() 
                     population.spread_of_disease() 
                     population.array_T.append(t)
                     t = t + 1
                array_S_total.append(population.array_S)
                array_I_total.append(population.array_I)
                array_R_total.append(population.array_R)
                rep = rep + 1

            array_means_S = np.mean(array_S_total, axis = 0)
            array_means_I = np.mean(array_I_total, axis = 0)
            array_means_R = np.mean(array_R_total, axis = 0)
            array_sd_S = np.std(array_S_total, axis = 0)
            array_sd_I = np.std(array_I_total, axis = 0)
            array_sd_R = np.std(array_R_total, axis = 0)
           
            name='Project1_probSus_'+str(probSus_vector[k])+'probInf_'+str(probInf_vector[m])
            
            """
            np.savetxt('array_means_S'+name, array_means_S)
            np.savetxt('array_means_I'+name, array_means_I)
            np.savetxt('array_means_R'+name, array_means_R)
            np.savetxt('array_S_total'+name, array_S_total)
            np.savetxt('array_I_total'+name, array_I_total)
            np.savetxt('array_R_total'+name, array_R_total)
            """

            # plot and save the means of the simulation with error bars that represent the 95% confidence interval 
            plt.gca().set_color_cycle(['green', 'blue', 'red'])
            plt.plot(population.array_T, array_means_S, label='S')
            plt.plot(population.array_T, array_means_I, label='I')
            plt.plot(population.array_T, array_means_R, label='R')
            plt.errorbar(population.array_T, array_means_S, yerr = 2*array_sd_S)
            plt.errorbar(population.array_T, array_means_I, yerr = 2*array_sd_I)
            plt.errorbar(population.array_T, array_means_R, yerr = 2*array_sd_R)
            plt.xlabel('Days')
            plt.ylabel('Population')
            plt.title('Deterministic model \n probSus:'+str(probSus_vector[k])+'; probInf:'+str(probInf_vector[m]))
            plt.legend(fontsize=8)
            plt.savefig(name+'.png', dpi=1200)

            # plot and save the 20-day period graph
            plt.clf()            
            plt.gca().set_color_cycle(['green', 'blue', 'red'])
            plt.plot(population.array_T, array_means_S, label='S')
            plt.plot(population.array_T, array_means_I, label='I')
            plt.plot(population.array_T, array_means_R, label='R')
            plt.xlim(0,20)
            plt.errorbar(population.array_T, array_means_S, yerr = 2*array_sd_S)
            plt.errorbar(population.array_T, array_means_I, yerr = 2*array_sd_I)
            plt.errorbar(population.array_T, array_means_R, yerr = 2*array_sd_R)
            plt.xlabel('Days')
            plt.ylabel('Population')
            plt.title('Deterministic model (20 days) \n probSus:'+str(probSus_vector[k])+'; probInf:'+str(probInf_vector[m]))
            plt.legend(fontsize=8)
            plt.savefig('20 days ' + name +'.png', dpi=1200)
    
class Population():
    
   def __init__(self):
       """
       Initial creation of the population where the dimensions are determined.
       """
       self.height = 50
       self.width = 50
       self.matrix = np.zeros((self.height,self.width))
       self.probSusceptible = 0.1
       self.probInfected = 0.1
       self.susceptible = [0]
       self.infected = [1,2]
       self.immune = [3,4,5,6,7]
       self.time = 100
       self.array_S = []
       self.array_I = []
       self.array_R = []
       self.array_T = []
       
   def initial_values(self):
       """
       Set the initial grid.
       """
       self.value=[0,1,2,3,4,5,6,7]
       self.prob=[self.probSusceptible, (1-self.probSusceptible)*self.probInfected+self.probSusceptible, 1]  
       self.matrix=[]
       for i in range(self.height):
           self.matrix.append([])
           for j in range(self.width):
               self.matrix[i].append(probability.initial(self.value,self.prob)) 

   def show(self):
       """
       Print grid to console and store the values for S, I and R per time step in arrays.
       """
       self.image = self.matrix
       self.cmap = mpl.colors.ListedColormap(['green', 'midnightblue', 'blue', 'darkred', 'red', 'tomato', 'salmon', 'lightsalmon'])
       self.bounds = [-1, 1, 2, 3, 4, 5, 6, 7, 8]
       self.norm = mpl.colors.BoundaryNorm(self.bounds, self.cmap.N)
       grid = plt.imshow(self.image, interpolation = 'nearest', cmap = self.cmap, norm = self.norm)
       cbar = plt.colorbar(grid, cmap=self.cmap, norm=self.norm, ticks=self.bounds, boundaries=self.bounds)
       labels = ['S', 'I day 1', 'I day 2', 'R day 1', 'R day 2', 'R day 3', 'R day 4', 'R day 5']
       cbar.set_ticklabels(labels)
       plt.show() 
       plt.clf()
    
       self.count_S = 0
       self.count_I = 0
       self.count_R = 0

       for i in range(self.height):
           for j in range(self.width):
               if self.matrix[i][j] in self.susceptible:
                   self.count_S = self.count_S + 1
               if self.matrix[i][j] in self.infected:
                   self.count_I = self.count_I + 1
               if self.matrix[i][j] in self.immune:
                   self.count_R = self.count_R + 1
       self.array_S.append(self.count_S)
       self.array_I.append(self.count_I)
       self.array_R.append(self.count_R)
            
   def spread_of_disease(self):
        """
        Examine state of the neighbors of each cell, and update new matrix for t+1. 

        """
        self.new_matrix = np.zeros((self.height, self.width))
        for i in range(self.height):
            for j in range(self.width):
                self.neighbors = []
                try:
                    self.neighbors.append(self.matrix[i+1][j])
                except IndexError:
                    pass
                if i-1 >= 0:
                    self.neighbors.append(self.matrix[i-1][j])
                try:
                    self.neighbors.append(self.matrix[i][j+1])
                except IndexError:
                    pass
                if j-1 >= 0:                    
                    self.neighbors.append(self.matrix[i][j-1])
                if self.matrix[i][j] == 0: 
                    for k in range(len(self.neighbors)):
                        if self.neighbors[k] in self.infected:
                            self.new_matrix[i][j] = 1
                            break
                elif self.matrix[i][j] > 0 and self.matrix[i][j] < 7:
                    self.new_matrix[i][j] = self.matrix[i][j] + 1
                else:
                    self.new_matrix[i][j] = 0
                    
        self.matrix = self.new_matrix
        
if __name__ == "__main__":
    main()

 