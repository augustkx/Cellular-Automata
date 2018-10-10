# -*- coding: utf-8 -*-
"""
@authors: Kaixin and Suzan

Stochastic SIR Model 3 model with Typhoid Mar, initially on one is sick

"""

import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mpl
import probability
    
def main():
    """
    Simulate the spread of disease n times, for a time period of population.time. 
    """
    # run through the different combinations of probabilities.
    array_S_total = []
    array_I_total = []
    array_R_total = []
    
    # simulate a hundred times        
    rep = 1    
    while rep <= 100:    
        population = Population() 
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
            
    name='Project5-4'
    '''
    np.savetxt('array_means_S'+name, array_means_S)
    np.savetxt('array_means_I'+name, array_means_I)
    np.savetxt('array_means_R'+name, array_means_R)
    np.savetxt('array_S_total'+name, array_S_total)
    np.savetxt('array_I_total'+name, array_I_total)
    np.savetxt('array_R_total'+name, array_R_total)
    '''
    # plot the means of the simulation with error bars that represent the 95% confidence interval 
    plt.gca().set_color_cycle(['green', 'blue', 'red'])
    plt.plot(population.array_T, array_means_S, label='S')
    plt.plot(population.array_T, array_means_I, label='I')
    plt.plot(population.array_T, array_means_R, label='R')
    plt.errorbar(population.array_T, array_means_S, yerr = 2*array_sd_S)
    plt.errorbar(population.array_T, array_means_I, yerr = 2*array_sd_I)
    plt.errorbar(population.array_T, array_means_R, yerr = 2*array_sd_R)
    plt.xlabel('Days')
    plt.ylabel('Population')
    plt.title('Stochastic model 3 with Typhoid Mary')
    plt.legend(fontsize=8)
    plt.savefig(name+'.png', dpi=1200)

    # plot the  20-day-period graph            
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
    plt.title('Stochastic model 3 with Typhoid Mary (20 days)' )
    plt.legend(fontsize=8)
    plt.savefig('shorter_period_'+name+'.png', dpi=1200)             
    
class Population():

    def __init__(self):
        """
        Initial creation of the population where the dimensions are determined.
        """
        self.height = 50
        self.width = 50
        self.matrix = np.zeros((self.height,self.width))
        #self.probSusceptible = 0.1
        #self.probInfected = 0.9
        self.susceptible = [0]
        self.infected = [1,2]
        self.immune = [3,4,5,6,7]
        self.time = 100
        self.array_S = []
        self.array_I = []
        self.array_R = []
        self.array_T = []   
        self.infected1 = [1]
        self.infected2 = [2]
        self.value1=[0,1] 
        self.MaryMatrix=[8]   
        
        #some varibles used when Mary moves: each represents: below, up, right, left.
        self.directionOfNeighbors=[[1,0],[-1,0],[0,1],[0,-1]]


    def initial_values(self):      
        '''
        Initially set Mary into one of the cells randomly
       
        '''
        self.RowOfMary=random.randint(0,self.height-1)
        self.ColumnOfMary=random.randint(0,self.width-1)              
        self.matrix[self.RowOfMary][self.ColumnOfMary] = 8    
    
    def show(self):
        """
        Print grid to console.
        """
        self.image = self.matrix
        self.cmap = mpl.colors.ListedColormap(['green', 'midnightblue', 'blue', 'darkred', 'red', 'tomato', 'salmon', 'lightsalmon', 'yellow'])
        self.bounds = [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.norm = mpl.colors.BoundaryNorm(self.bounds, self.cmap.N)
        grid = plt.imshow(self.image, interpolation = 'nearest', cmap = self.cmap, norm = self.norm)
        cbar = plt.colorbar(grid, cmap=self.cmap, norm=self.norm, ticks=self.bounds, boundaries=self.bounds)
        labels = ['S', 'I day 1', 'I day 2', 'R day 1', 'R day 2', 'R day 3', 'R day 4', 'R day 5', 'Mary']
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
                    self.neighbors.append('non')
                               
                if i-1 >= 0:
                    self.neighbors.append(self.matrix[i-1][j])
                else:
                    self.neighbors.append('non')
                
                try:
                    self.neighbors.append(self.matrix[i][j+1])
                except IndexError:
                    self.neighbors.append('non')
               
                if j-1 >= 0:                    
                    self.neighbors.append(self.matrix[i][j-1])
                else:
                    self.neighbors.append('non')
                    
                # mark the neighbor matrix for Mary    
                if i==self.RowOfMary and j==self.ColumnOfMary:
                    self.neighborOfMary=self.neighbors

                if self.matrix[i][j] == 0: 
                    self.level=0
                    for k in range(len(self.neighbors)):
                        if self.neighbors[k] in self.infected1 or self.neighbors[k] in self.MaryMatrix:
                            self.level +=1
                        elif  self.neighbors[k] in self.infected2:
                            self.level +=0.5
                    if  self.level>0:
                        self.probCatch=float(self.level)/(len(self.neighbors)-self.neighbors.count('non'))
                        self.prob1=[self.probCatch,1]
                        self.new_matrix[i][j] =  probability.ProbCatchF(self.value1,self.prob1)
                            
                elif self.matrix[i][j] > 0 and self.matrix[i][j] < 7:
                    self.new_matrix[i][j] = self.matrix[i][j] + 1
 
                elif self.matrix[i][j]==8:
                    self.new_matrix[i][j]=8
                else:
                    self.new_matrix[i][j] = 0
    
        self.matrix = self.new_matrix 

        # Mary changes her position with one of her neighbors
        self.direction=random.randint(0,3)
        if self.neighborOfMary[self.direction]=='non':
            while  self.neighborOfMary[self.direction] =='non':
                self.direction=random.randint(0,3) 
        self.move=self.directionOfNeighbors[self.direction]
        self.matrix[self.RowOfMary][self.ColumnOfMary], self.matrix[self.RowOfMary+self.move[0]][self.ColumnOfMary+self.move[1]]= self.matrix[self.RowOfMary+self.move[0]][self.ColumnOfMary+self.move[1]],self.matrix[self.RowOfMary][self.ColumnOfMary]          
        self.RowOfMary=self.RowOfMary+self.move[0]
        self.ColumnOfMary=self.ColumnOfMary+self.move[1]
        
if __name__ == "__main__":
    main()

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 