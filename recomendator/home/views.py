from __future__ import print_function
from __future__ import division
import collections
from os import name
from django.shortcuts import render
import math
import random
import time
import string
import urllib.request
import itertools
#import requests
import json 
import pandas as pd
from pandas import DataFrame
from k_means_constrained import KMeansConstrained
import numpy as np
from haversine import haversine, Unit
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from geojson import Feature, Point, FeatureCollection


# Create your views here.
def home_view(request, *args,**kwargs):
    return render(request,"home.html",{})
#Begins own functions

#HELD KARP function
def held_karp(n_held_places,dists):
    """
    Implementation of Held-Karp, an algorithm that solves the Traveling
    Salesman Problem using dynamic programming with memoization.
    Parameters:
        dists: distance matrix
    Returns:
        A tuple, (cost, path).
    """
    n = n_held_places

    # Maps each subset of the nodes to the cost to reach that subset, as well
    # as what node it passed before reaching this subset.
    # Node subsets are represented as set bits.
    C = {}

    # Set transition cost from initial state
    for k in range(1, n):
        C[(1 << k, k)] = (dists[0][k], 0)

    # Iterate subsets of increasing length and store intermediate results
    # in classic dynamic programming manner
    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            # Set bits for all nodes in this subset
            bits = 0
            for bit in subset:
                bits |= 1 << bit

            # Find the lowest cost to get to this subset
            for k in subset:
                prev = bits & ~(1 << k)

                res = []
                for m in subset:
                    if m == 0 or m == k:
                        continue
                    res.append((C[(prev, m)][0] + dists[m][k], m))
                C[(bits, k)] = min(res)

    # We're interested in all bits but the least significant (the start state)
    bits = (2**n - 1) - 1

    # Calculate optimal cost
    res = []
    for k in range(1, n):
        res.append((C[(bits, k)][0] + dists[k][0], k))
    opt, parent = min(res)

    # Backtrack to find full path
    path = []
    for i in range(n - 1):
        path.append(parent)
        new_bits = bits & ~(1 << parent)
        _, parent = C[(bits, parent)]
        bits = new_bits

    # Add implicit start state
    path.append(0)

    return opt, list(reversed(path))

#ANS functions
def make_cities(number_of_cities):
    return [a for a in range(number_of_cities)]
def make_dictionary(data,cluster_local):
    lista_nombres = list()
    for i in range(len(cluster_local)):
        nombre =data[(data['Latitud'] == cluster_local[i][0]) & (data['Longitud'] == cluster_local[i][1])]['Nombre'].tolist()
        lista_nombres.append(nombre[0])
    return {a:lista_nombres[a] for a in range(len(lista_nombres))}
def pretty_print(matrix,dictionary):
    pretty = DataFrame(matrix)
    pretty.columns = dictionary.values()
    pretty.index = dictionary.values()
    print(pretty)
    print("")
def print_path(path_traveled,dictionary,i):
    print("Ant ",i+1,": ",dictionary[path_traveled[0]],end="")
    for i in range(len(dictionary)-1):
        print(" - ",end="")
        print(dictionary[path_traveled[i+1]],end="")
    print("")
def print_cost(all_paths,dictionary,i,cost):
    print("Ant ",i+1,"( ",dictionary[all_paths[i][0]],end="")
    for j in range(len(dictionary)-1):
        print(" - ",end="")
        print(dictionary[all_paths[i][j+1]],end="")
    print(") - Cost: ",cost)
def print_global(best_global_path,dictionary):
    print("best_global_path :",dictionary[best_global_path[0][0]],end="")
    for i in range(len(dictionary)-1):
        print(" - ",end="")
        print(dictionary[best_global_path[0][i+1]],end="")
    print(" - Cost: ",best_global_path[1])
def fitness(all_paths,distances,n_ants,cities,dictionary,q):
    my_fitness = []
    for i in range(n_ants):
        cost = 0
        for j in range(cities-1):  
            cost+= q * distances.item(  (    all_paths[i][j]  ,  all_paths[i][j+1]   )   )
        my_fitness.append(cost)
        #print_cost(all_paths,dictionary,i,cost)
    return my_fitness[:]
def intensification(visited_cities,pheromone,visibility,alpha,beta,phi,local_initial_city,calculations_vector,dictionary):
    for j in range(len(visited_cities)):
        #print("Key1:",local_initial_city,"Key2:",visited_cities[i])
        pheromone_distance = math.pow( pheromone.item( (local_initial_city,visited_cities[j]) ) ,alpha   )
        visibility_distance = math.pow( visibility.item( (local_initial_city,visited_cities[j]) ) ,  beta   )
        calculations_vector.append(pheromone_distance*visibility_distance)
        #print(dictionary[local_initial_city],"-",dictionary[visited_cities[j]],": t = ",pheromone_distance,"n = ",visibility_distance,"t*n = ",pheromone_distance*visibility_distance)
    next_city = visited_cities[calculations_vector.index(max(calculations_vector))]
    #print("Next City: ",dictionary[next_city])
    
    pheromone_local_update = ( (1-phi)*pheromone.item( (local_initial_city,next_city) ) ) + phi*0.1
    #print("Updating the arc: ",dictionary[local_initial_city],"-",dictionary[next_city],"(v): (1-e)*",pheromone.item( (local_initial_city,next_city) ),"+ e*0.1 = ", pheromone_local_update)
    pheromone.itemset( (local_initial_city,next_city), pheromone_local_update )
    #print("")
    return next_city
def diversification(visited_cities,pheromone,visibility,alpha,beta,phi,local_initial_city,calculations_vector,dictionary):
    for j in range(len(visited_cities)):
        #print("Key1:",local_initial_city,"Key2:",visited_cities[i])
        pheromone_distance = pow( pheromone.item( (local_initial_city,visited_cities[j]) ) ,alpha   )
        visibility_distance = pow( visibility.item( (local_initial_city,visited_cities[j]) ) ,  beta   )
        calculations_vector.append(pheromone_distance*visibility_distance)
        #print(dictionary[local_initial_city],"-",dictionary[visited_cities[j]],": t = ",pheromone_distance,"n = ",visibility_distance,"t*n = ",pheromone_distance*visibility_distance)
    sum_calculations = sum(calculations_vector)
    odds_vector = []
    #print("Sum: ",sum_calculations)
    for j in range(len(calculations_vector)):
        odds_vector.append(calculations_vector[j]/sum_calculations)
        #print(dictionary[local_initial_city],"-",dictionary[visited_cities[j]],": prob =",odds_vector[j])
    random_election = random.uniform(0,1)
    #print("Random number for probability: ",random_election)
    roulette = 0
    for j in range(len(odds_vector)):
        roulette+= odds_vector[j]
        if(random_election<roulette):
            elected_city = visited_cities[j]
            break
    #print("next city ",dictionary[elected_city])
    pheromone_local_update = ( (1-phi)*pheromone.item( (local_initial_city,elected_city) ) ) + phi*0.1
    #print("Updating the arc: ",dictionary[local_initial_city],"-",dictionary[elected_city],"(v): (1-e)*",pheromone.item( (local_initial_city,elected_city) ),"+ e*0.1 = ", pheromone_local_update)
    pheromone.itemset( (local_initial_city,elected_city), pheromone_local_update )

    #print("")
    return elected_city
def ants_walking(n_ants,distances,visibility,pheromone,cities,dictionary,alpha,beta,rho,q,q0,phi,initial_city,best_global_path):
    all_paths=[]
    all_fitness=[]
    for i in range(n_ants):
        #print("Ant ",i+1)
        local_initial_city =  initial_city
        #print("Initial City: ",dictionary[local_initial_city])
        visited_cities = cities[:]
        path_traveled = []
        path_traveled.append(local_initial_city)
        while(len(visited_cities)!=1):
            visited_cities.remove(local_initial_city)
            calculations_vector = []
            q0_probability = random.uniform(0,1)
            #print("probability of q: ",q0_probability)
            if(q0_probability<q0):   
                #print("Intensification Tour")
                local_initial_city = intensification(visited_cities,pheromone,visibility,alpha,beta,phi,local_initial_city,calculations_vector,dictionary)
            elif(q0_probability>q0):
                #print("Diversification Tour")
                local_initial_city = diversification(visited_cities,pheromone,visibility,alpha,beta,phi,local_initial_city,calculations_vector,dictionary)
            path_traveled.append(local_initial_city)
        #print_path(path_traveled,dictionary,i)
        all_paths.append(path_traveled)
    #print("")
    all_fitness = fitness(all_paths,distances,n_ants,len(cities),dictionary,q)
    best_local_path = [all_paths[all_fitness.index(min(all_fitness))],all_fitness[all_fitness.index(min(all_fitness))]]
    if(len(best_global_path[0])==0):
        best_global_path = best_local_path
    else:
        if(best_local_path[1]<best_global_path[1]):
            best_global_path = best_local_path
    #print("------------")
    #print_global(best_global_path,dictionary)
    #print("------------")

    flag = 0
    for i in range(len(cities)):
        for j in range(len(cities)):
            if(j<i or i==j):
                continue
            #updating_value = rho*pheromone.item((i,j)) teacher correction
            updating_value = pheromone.item((i,j))
            #print(dictionary[i],"-",dictionary[j],": pheromone = ",updating_value,end="")     
            for k in range(len(best_global_path[0])-1):
                #i just need to search  and compare [i]  [j]    to   [k][l]    [k][l+1] 
                if(best_global_path[0][k] == i and best_global_path[0][k+1] == j):
                    updating_value *= rho
                    updating_value += (1./best_global_path[1]) 
                    flag =1                    
                elif(best_global_path[0][k] == j and best_global_path[0][k+1] == i):
                    updating_value *= rho
                    updating_value += (1./best_global_path[1])
                    flag =1
                #print(" + ",end="")
                if(flag==1):
                    #print(" + ",1./best_global_path[1],end="")
                    flag = 0
                elif(flag==0):
                    #print(" + 0.0 ",end="")
                    flag = 2
            #print(" = ",updating_value)
            pheromone.itemset((i,j),updating_value)
            pheromone.itemset((j,i),updating_value)
    #print("Distance Matrix: ")
    #pretty_print(distances,dictionary)
    #print("Visibility Matrix: ")
    #pretty_print(visibility,dictionary)
    #print("Final Local pheromone matrix")
    #pretty_print(pheromone,dictionary)   
    return best_global_path
def ACS(distanceMatrix,local_data,cluster_local):
    #parameters
    n_ants = 8
    initial_city = 1
    initial_pheromone = 0.1
    alpha = 1.0
    beta = 1.0
    rho = 0.01
    q = 1.0
    q0 = 0.7
    phi = 0.05
    n_iteration = 50
    #help parameters
    cities = make_cities(distanceMatrix.shape[0])
    best_global_path = [[]]
    dictionary_names = make_dictionary(local_data,cluster_local)
    #Visibility of cities
    visibility = np.divide(1,distanceMatrix,out=np.zeros_like(distanceMatrix), where=distanceMatrix!=0)
    
    #pheromone matrix initial
    pheromone = np.copy(visibility)
    pheromone.fill(initial_pheromone)
    np.fill_diagonal(pheromone,0.)

    #Printing Values
    print("Parameters:")
    print("Number of Ants = ",n_ants)
    print("Initial Feromone = ",initial_pheromone)
    print("Alpha = ",alpha)
    print("Beta = ",beta)
    print("Rho = ",rho)
    print("Q = ",q)
    print("q0 = ",q0)
    print("e(Phi) = ",phi)
    print("Number of Iterations = ",n_iteration)

    #Printing Matrixes
    print("Distance Matrix: ")
    pretty_print(distanceMatrix,dictionary_names)
    print("Visibility Matrix: ")
    pretty_print(visibility,dictionary_names)
    print("Pheromone Matrix")
    pretty_print(pheromone,dictionary_names)

    #ACS
    for i in range(n_iteration):
        #print("---------------------------------")
        #print("Iteration",i+1)
        #print("visibility Matrix: ")
        #pretty_print(visibility,dictionary_names)
        #print("Pheromone Matrix")
        #pretty_print(pheromone,dictionary_names)
        best_global_path = ants_walking(n_ants,distanceMatrix,visibility,pheromone,cities,dictionary_names,alpha,beta,rho,q,q0,phi,initial_city,best_global_path)
        #print("Total Iteration: ",i+1)
        #print("------------")
        #print_global(best_global_path,dictionary_names)
        #print("------------")
    return best_global_path,dictionary_names

#CLONALG functions
def make_pair_random(n_pairs,size):
    list_of_pairs = []
    for i in range(n_pairs):
        r1 = random.randint(0,size)
        r2 = random.randint(0,size)
        if(r1==r2):
            r2 = random.randint(0,size)
        list_of_pairs.append([r1,r2])
    return list_of_pairs
def init_population(distances,dictionary,population_p):
    population = []
    costs = []
    for i in range(population_p):
        paths = list(range(0,distances.shape[0]))
        random.shuffle(paths)
        population.append(paths)
        cost = 0.
        #print(i+1,")",sep="",end="")
        #for element in paths:
            #print(dictionary[element],end="")
        for j in range(distances.shape[0]-1):
            #cost += distances[paths[i]][paths[i+1]]
            cost += distances.item((  paths[j] , paths[j+1]  ))
        costs.append(cost)
        #print("\t",cost)
    #print("")
    return population,costs
def make_populationF(population_f,population,costs,dictionary):
    copy_population = population[:]
    copy_costs = costs[:]
    population_f_list = []
    index = sorted(range(len(copy_costs)), key=lambda k: copy_costs[k])[:population_f ]
    for i in range(len(index)):
        population_f_list.append(copy_population[index[i]])
        #print(i+1,")",sep="",end="")
        #for element in population_f_list[i]:
            #print(dictionary[element],end="")
        #print("\t",copy_costs[index[i]])
    #print("")
    return population_f_list
def make_populationPClone(populationf,dictionary,population_pclone):
    n_clone = population_pclone
    flag=0
    populationPclone = []
    populationf_copy = populationf[:]
    while(n_clone!=0):
        for i in range(n_clone):
            populationPclone.append(populationf_copy[flag])
        n_clone-=1
        flag+=1

    #for i in range(len(populationPclone)):
        #print(i+1,") ",sep="",end="")
        #for element in populationPclone[i]:
            #print(dictionary[element],end="")
        #print("")
    #print("")
    return populationPclone
def make_populationPHyper(populationPclone,dictionary,distances):
    populationPclone_copy = populationPclone[:]
    populationPhyper = []
    newcosts = []
    pair = 1
    flag = 1
    aux = 5
    for i in range(len(populationPclone_copy)):
        cost = 0
        if(flag > aux):
            flag = 1
            aux -= 1
            pair += 1
        randoms = make_pair_random(pair,len(populationPclone_copy[i])-1)
        for element in randoms:
            populationPhyper.append(populationPclone_copy[i][:])
            populationPhyper[i][element[0]],populationPhyper[i][element[1]] = populationPhyper[i][element[1]],populationPhyper[i][element[0]]
        #print(i+1,")",sep="",end=" ")
        #for element2 in populationPclone_copy[i]:
            #print(dictionary[element2],sep="",end="")
        #print("\t",randoms,"\t",sep="",end="")
        #for element3 in populationPhyper[i]:
            #print(dictionary[element3],sep="",end="")
        for j in range(distances.shape[0]-1):
            cost += distances.item(( populationPhyper[i][j]  , populationPhyper[i][j+1]  ))
        newcosts.append(cost)
        #print("\t",cost)    
        flag += 1
    return populationPhyper[:],newcosts[:]
def make_populationS(populationPhyper,newcosts,dictionary,distances,population_s):
    populationPhyper_copy = populationPhyper[:]
    newcosts_copy = newcosts[:]
    populationS = []
    costsS = []
    index = sorted(range(len(newcosts_copy)), key=lambda k: newcosts_copy[k])[:population_s ]
    for i in range(len(index)):
        populationS.append(populationPhyper_copy[index[i]])
        #print(i+1,")",sep="",end="")
        #for element in populationS[i]:
            #print(dictionary[element],end="")
        #print("\t",newcosts_copy[index[i]])
        costsS.append(newcosts_copy[index[i]])
    #print("")
    return populationS[:],costsS[:]
def make_populationR(distances,dictionary,population_r):
    populationR = []
    costsR = []
    for i in range(population_r):
        paths = list(range(0,distances.shape[0]))
        random.shuffle(paths)
        populationR.append(paths)
        cost = 0.
        #print(i+1,")",sep="",end="")
        #for element in paths:
            #print(dictionary[element],end="")
        for j in range(distances.shape[0]-1):
            cost += distances.item((  paths[j] , paths[j+1]  ))
        costsR.append(cost)
        #print("\t",cost)
    #print("")
    return populationR[:],costsR[:]
def selectingbests(population,costs,populationS,costsS,populationR,costsR,distances,dictionary,nbests,population_p):
    newpopulation = []
    newcosts = []
    populationp_copy = population[:]
    populations_copy = populationS[:]
    populationr_copy = populationR[:]
    costsp_copy = costs[:]
    costss_copy = costsS[:]
    costsr_copy = costsR[:]

    index = sorted(range(len(costsp_copy)), key=lambda k: costsp_copy[k])[:nbests ]
    for item in index:
        newpopulation.append(  populationp_copy[item][:]  )
        newcosts.append(  costsp_copy[item]  )
    
    populations_copy += populationr_copy
    costss_copy += costsr_copy
    aux = population_p-nbests
    index1 = sorted(range(len(costss_copy)), key=lambda k: costss_copy[k])[:aux ]
    for item1 in index1:
        newpopulation.append(  populations_copy[item1][:]  )
        newcosts.append(  costss_copy[item1]  )

    #for i in range(len(newpopulation)):
        #print(i+1,")",sep="",end="")
        #for obj in newpopulation[i]:
            #print(dictionary[obj],sep="",end="")
        #print("\t",newcosts[i])
    return newpopulation[:],newcosts[:]
def CLONALG(distanceMatrix,local_data,cluster_local):
    #parameters
    population_p = 5
    population_f = 4
    population_pclone = 4
    population_phyper = 4
    population_s = 5
    population_r = 2
    antibody = 8
    n_iteration = 200
    nbests = 2

    #printing parameters
    print("-population size P =",population_p)
    print("-population size F =",population_f)
    print("-population size PClone y PHyper =",population_pclone)
    print("-population size S =",population_s)
    print("-population size R =",population_r)
    print("-vector size(antibody) =",antibody)
    print("-number of iterations =",n_iteration)

    #auxiliar variables
    clonalg_dictionary_names = make_dictionary(local_data,cluster_local)
    
    ###inital population
    #print("*** Population P ***")
    population,costs = init_population(distanceMatrix,clonalg_dictionary_names,population_p)

    #begin iteration
    for i in range(n_iteration):
        #print("*** Iteration ",i+1," ***")
        #print("*** Population F ***")
        populationf = make_populationF(population_f,population,costs,clonalg_dictionary_names)
        #print("*** Population PClone ***")
        populationPclone = make_populationPClone(populationf,clonalg_dictionary_names,population_pclone)
        #print("*** Population PHyper ***")
        populationPhyper,newcosts = make_populationPHyper(populationPclone,clonalg_dictionary_names,distanceMatrix)
        #print("*** Population S ***")
        populationS,costsS = make_populationS(populationPhyper,newcosts,clonalg_dictionary_names,distanceMatrix,population_s)
        #print("*** Population R ***")
        populationR,costsR = make_populationR(distanceMatrix,clonalg_dictionary_names,population_r)
        #print("*** Population P ***")
        population,costs = selectingbests(population,costs,populationS,costsS,populationR,costsR,distanceMatrix,clonalg_dictionary_names,nbests,population_p)
    return population,costs,clonalg_dictionary_names

#Making distances
def make_matriz_distancia1(mycluster):
    hardcopy_points = mycluster[:]
    matriz_distancia = np.zeros(shape=(len(hardcopy_points),len(hardcopy_points)))
    #print(matriz_distancia,"este",len(mycluster),"len")
    vector_aux = np.zeros(len(hardcopy_points))
    for row in range(len(hardcopy_points)):
        for col in range(len(hardcopy_points)):
            #print(hardcopy_points[row],"row",hardcopy_points[col],"col")
            distancia_simple = haversine(tuple(hardcopy_points[row]),tuple(hardcopy_points[col]))
            #print("tupla1",tuple(hardcopy_points[row]),"tupla2",tuple(hardcopy_points[col]),distancia_simple )
            vector_aux[col] = distancia_simple
        #METER LOS DATOS EN LA MATRIZ ASI MATRIZ[ROW] = ARRAY QUE SE HALLO ARRIBA
        matriz_distancia[row] = vector_aux[:]
    return matriz_distancia

def make_matriz_distancia(mycluster):
    #copy of matrix
    hardcopy_points = mycluster[:]
    #do container for the distances
    matriz_distancia = np.zeros(shape=(len(hardcopy_points),len(hardcopy_points)))
    vector_aux = np.zeros(len(hardcopy_points))

    #parameters for the requests
    API_KEY = "AIzaSyBzv_L64hctIEnbvd5M0igrUpMLWvt_aFA"
    urlmask = "https://maps.googleapis.com/maps/api/distancematrix/json?origins="
    destmask = "&destinations="
    coma=","
    finalurlmask="&key="

    #get every distance a save in the distance matrix
    for row in range(len(hardcopy_points)):
        for col in range(len(hardcopy_points)):
            myorigin = tuple(hardcopy_points[row])
            originlat = str(myorigin[0])
            originlong = str(myorigin[1])
            
            mydest = tuple(hardcopy_points[col]) 
            destlat = str(mydest[0])
            destlong = str(mydest[1])

            finalurl = urlmask + originlat + coma + originlong +destmask + destlat + coma + destlong + finalurlmask + API_KEY

            response = urllib.request.urlopen(finalurl)
            responseJson = json.loads(response.read())
            
            print(responseJson," JSON ", hardcopy_points[row],"inicio",hardcopy_points[col],"destino","\n")
            real_distance_google = responseJson['rows'][0]['elements'][0]['distance']['value']

            
            vector_aux[col] = real_distance_google
        #METER LOS DATOS EN LA MATRIZ ASI MATRIZ[ROW] = ARRAY QUE SE HALLO ARRIBA
        matriz_distancia[row] = vector_aux[:]
    return matriz_distancia



def make_itinearary(dias,lugares):
    #Data to return
    times = list()
    all_recomendation_name = list()
    all_recomendation_geo = list()
    #data raw
    data = pd.read_csv("home/data/base_de_datos_geograficos_turisticos_refined.csv")
    #data = pd.read_csv("D:/tesis/Recomendador-de-POI-s-con-clusterizacion-equilibrada-y-algoritmos-bioinspirados/recomendator/home/prueba_datos.csv")
    #data = pd.read_csv("home/data/base_de_datos_geograficos_turisticos.csv")
    #mi_localizacion =  [-16.412828, -71.517236]
    clusters_dict = dict()
    #converting data to use
    df = pd.DataFrame(data)
    x = df['Latitud'].values
    y = df['Longitud'].values
    #making a matrix of data
    X = np.array(list(zip(x,y)))
    #print("DATA",X)
    #input of days  TO CHANGE MAYBE
    n_days = dias
    n_places = lugares
    #number of clusters, getting labels
    #time function
    clustering_time = time.process_time()
    kmeans = KMeansConstrained(
        n_clusters=n_days,
        size_min=n_places,
        size_max=((len(data)-1)/n_days)+1,
        random_state=0
    )
    #fixed k-means
    kmeans.fit_predict(X)
    centroid, labels = kmeans.cluster_centers_,kmeans.labels_
    final_clustering_time = time.process_time() - clustering_time

    ###print(kmeans1.cluster_centers_,"FIXED centers")
    ###print(kmeans1.labels_,"FIXED labels")

    #k-means normal
    """
    kmeans = KMeans(n_clusters=n_days)
    kmeans = kmeans.fit(X)
    labels,centroid = kmeans.predict(X), kmeans.cluster_centers_
    print(type(centroid))
    print(type(labels))
    """
    #colors for plotting
    colors = ["m.","r.","c.","y.","b."]

    #plotting

    for i in range(len(X)):
        #print("Coordenadas: ",X[i],"Label Cluster: ",labels[i])
        #add data in cluster dicts
        if labels[i] in clusters_dict:
            indice = int(labels[i])
            clusters_dict[indice].append(X[i])
        else:
            indice = int(labels[i])
            clusters_dict[indice] = list([X[i]])
        plt.plot(X[i][0],X[i][1], colors[labels[i]],markersize=10)
    plt.scatter(centroid[:,0],centroid[:,1],marker="X",s=150,linewidths=5,zorder=10)
    #plt.show()
    """
    print(clusters_dict[1])
    for a in clusters_dict[1]:
        print(a[0],a[1])
    """
    n_iteration=1
    #print(clusters_dict)

    #Calculate ACS for every cluster
    #print("recomendation using ACS")
    # for cluster in clusters_dict:
    #     ###slice matrix to n number of POIs user wanted
    #     slice_distance = clusters_dict[cluster][0:n_places]

    #     #make distance matrix
    #     distancia = make_matriz_distancia(slice_distance)

    #     # #get best global path
    #     cluster_global_path_ACS,cluster_dictionary_names_ACS = ACS(distancia,data,slice_distance)#clusters_dict[cluster])
    #     # #show best global path
    #     print("For the cluster ",n_iteration)
    #     print_global(cluster_global_path_ACS,cluster_dictionary_names_ACS)
    #     n_iteration+=1
    # print("HELD KARP processing time")
    # for cluster in clusters_dict:
    #     ###slice matrix to n number of POIs user wanted
    #     slice_distance = clusters_dict[cluster][0:n_places]
    #     #make distance matrix
    #     distancia = make_matriz_distancia(slice_distance)
    #     #take times
    #     held_time = time.process_time()
    #     #held_karp tranversing min time
    #     mintime_to_transverse = held_karp(n_places,distancia)
    #     held_final_time = time.process_time() - held_time
    #     #final time
    #     held_processing_time = held_final_time + final_clustering_time
    #     print("el menor tiempo para recorrer es ",mintime_to_transverse, " el tiempo de procesamiento es ", held_processing_time)
    #     n_iteration+=1
    print("recomendation using CLONALG")
    n_iteration=1
    #Calculate CLONALG for every cluster
    for cluster in clusters_dict:
        #partial data
        partial_data = list()
        ###slice matrix to n number of pois user wanted
        slice_distance = clusters_dict[cluster][0:n_places]

        #make distance matrix
        distancia = make_matriz_distancia(slice_distance)


        #take time
        clonalg_time = time.process_time()
        #get best global path 
        clonalg_global_population, clonalg_global_costs, clonalg_global_dictionary_names =CLONALG(distancia,data,slice_distance)#clusters_dict[cluster])
        clonalg_final_time = time.process_time() - clonalg_time
        processing_time = clonalg_final_time + final_clustering_time
        
        
        
        #show best global path
        print("For the cluster ",n_iteration)
        for i in range(len(clonalg_global_population)):
            print(i+1,")",sep="",end="")
            for obj in clonalg_global_population[i]:
                print(clonalg_global_dictionary_names[obj],sep="",end=" - ")
            print("\t",clonalg_global_costs[i])
        n_iteration+=1
        print(processing_time,"seconds")

        #data to return
        times.append(processing_time)
        for obj in clonalg_global_population[0]:
            partial_data.append(clonalg_global_dictionary_names[obj])
            all_recomendation_name.append(clonalg_global_dictionary_names[obj])
        for i in range(len(partial_data)):
            lat = data[(data['Nombre'] == partial_data[i])]['Latitud'].tolist()
            long = data[(data['Nombre'] == partial_data[i])]['Longitud'].tolist()
            all_recomendation_geo.append([long,lat])

        
    return all_recomendation_name,all_recomendation_geo,times, clonalg_global_costs[0]
        

#END of own functions

#my controller

def recomendacion(request):
    n_dias = int(request.GET['dias'])
    n_lugares = int(request.GET['lugares'])
    names,geo,all_times,costs = make_itinearary(n_dias,n_lugares)
    list_features = list()
    cluster_counter = 0
    for i in range(len(names)):
        if(i%n_lugares==0):
            cluster_counter+=1
            print(i,"DEBUGGING")
        cluster_counter_property = "recorrido " + str(cluster_counter)
        feature = Feature(geometry=Point(geo[i]))
        feature.properties['nombre'] = names[i]
        feature.properties['coordenadas'] = geo[i]
        feature.properties['cluster'] = cluster_counter_property
        list_features.append(feature)
    feature_collection = FeatureCollection(list_features)
    print(feature_collection)
    mapbox_access_token = 'pk.eyJ1Ijoib25seWxvb2tlciIsImEiOiJja3dyemM5dDAxMWFsMm9zNmw1ZmhvYmRyIn0.IOeC5MgHLqx-bX09ispxkA'
    return render(request,"recomendacion.html",{"geojson":feature_collection,"tiempos":all_times,"n_items":n_lugares,"n_clusters":n_dias,"costs":costs,'mapbox_access_token': mapbox_access_token})