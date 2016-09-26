'''
Team: anahar-vrvernek-skariyat

Abstraction
    initial State: Start City with miles, time as 0
    Goal State: Destination City with calculated miles and time
    State Space: The entire map of US, Canada and Mexico
    Successor Function: depending on the algorithm and the option used, we would travel to one of the neighboring cities
    Cost: Cost function depends on the routing option used.i
            Distance : miles between the cities
            Time: time required to travel between the cities
            Scenic: speed for each path taken
            Segment: number of segment along the path

The road-segments.txt is read in a dictonary (graph) to create an adjaceny list
city-gps.txt is read in a dictionary (gps).

1. Which search algorithm seems to work best for each routing options?
Ans:
    1. Segments
        BFS and Astar produces the optimal path but Astar needs slightly less time as compared to BFS and DFS. The time difference increases as the distance between the cities increases. It also depends on the Heuristic used.
    2. Distance
        None of the algorithms seem to produce an optimal path. But when compared amongst themselves, Astar gives the minimum distance
    3. Time
        None of the algorithms seem to produce an optimal path.
    4. Scenic
        None of the algorithms seem to produce an optimal path.
(Regardless of the routing option use, BFS will give the same path since it is a blind search algo. Same for IDS and DFS)
        
2. Which algorithm is fastest in terms of the amount of computation time required by your program, and by how much, according to your experiments?
Ans:
    Astar runs fastest amongst other algorithms. Speed of Astar depends on the heuristic used. Better the heuristic, better time we get.
    Sample run for Bloomington,_Indiana to Denver,_Colorado in a loop of 100 and an average was removed
    (Time is same for all routing option in case of BFS DFS and IDS)
        BFS 0.02302 sec
        DFS 0.03412 sec
        IDS 3.439 sec
        Astar
            Segments 0.6769 sec
            Distance 0.0121 sec
            Time     0.0134 sec
            Scenic   0.0087 sec

3. Which algorithm requires the least memory, and by how much, according to your experiments? 
Ans:
    IDS requires the least amount of memory since it explores in depth first manner but until a certain height. DFS can pick up the wrond path for exploration and won't realize until it has reached the very end. whereas IDS explore only till the depth at which the solution is present making it better than DFS. Astar may keep on exploring multiplt branches based on the heuristic function applied
    
4. Which heuristic function did you use, how good is it, and how might you make it better?
Ans:
    We have used "haversine distance" to calculate the distance between current node and the goal node based on latitude and longitude. If lat and lon for a city/junction is not provided, we try to find next level of connected cities which has gps value populated for them and haversine distance is calculated from those cities to the destination city

    a.  For Astar - distance
            base Heuristic is used as the haversine distance and the distance covered so far, both are in miles
    b. Astar - segments
            we have considered the smallest distance as the minimum Segment length and then each time the heuristic is calculated by dividing minimumSegment / haversineDistance
    c.  Astar - time
            the haversineDistance is divided by maximum speed present in the road-segment.txt file. We were planning to take average speed but that would have overestimated in some scenarios
    d.  Astar - scenic
            A penalty is placed on the route each time a highway (speed > 55) is visited. the penalty is made variable based on the speed. It is calulated as haversineDistance * (speed - 54). 54 is taken since highway is considered above 55 miles/hour

we have used maximum speed and minimum segments in calutating heuristic function which is not ideal values. In order to improve the heuristic, we could use the actual values which would provide a more better estimation

5. Supposing you start in Bloomington, which city should you travel to if you want to take the longest possible drive (in miles) that is still the shortest path to that city? (In other words, which city is furthest from Bloomington?)

Ans:
   Skagway,_Alaska is the city that should be travelled if we want to take the longest possible drive (in miles) that is still the shortest path to that city. Used dijkstra algorithm to find the city.

'''

import sys
import time

global startCity
global endCity
global routingOption
global routingAlgo

#Read command line arguments
def readArg():
    global startCity
    global endCity
    global routingOption
    global routingAlgo

    if len(sys.argv) != 5:
        print "Enter valid number of arguments"
        return 1

    startCity = sys.argv[1]
    endCity = sys.argv[2]
    routingOption = sys.argv[3]
    routingAlgo = sys.argv[4]

#Read road-segments.txt
def readRoadSegments():
    f = open ("road-segments.txt","r")
    graph = {}
    overallDistance = 0.0
    overallTime = 0
    minSegment = 1000
  
    for line in f.readlines():
        word = line.split()

        #handle missing speed case
        if(len(word) < 5 ):
            word.insert(3,'50')

        #handle miles = 0 case
        if(int(word[2]) == 0):
            word[2] = '50'

        #handle speed = 0 case
        if(int(word[3]) == 0):
            word[3] = '50'

        if(minSegment > int(word[2])):
            minSegment = int(word[2])

        overallDistance += float(word[2])
        time = float(word[2]) / float(word[3])
        overallTime += time

        #Add path from source city to destination city in graph
        if word[0] not in graph:
            graph[word[0]] = [word[1:]]
        else:
            graph[word[0]] += [word[1:]]

        #Add path from destination city to source city in graph
        if word[1] not in graph:
            temp = [word[0]]
            temp += word[2:]
            graph[word[1]] = [temp]

        else:
            temp = [word[0]]
            temp += word[2:]
            graph[word[1]] += [temp]

    f.close()
    return (graph, minSegment, overallDistance, overallTime)

#Read city-gps.txt
def readCityGps():
    f = open ("city-gps.txt","r")
    graph = {}
    graph = {line.split()[0] : line.split()[1:] for line in f.readlines() if line.split()[0] not in graph}
    f.close()
    return graph

def checkInput(graph):
    if startCity not in graph:
        print "Please enter valid start city...exiting"
        return 1

    if endCity not in graph:
        print "Please enter valid end city...exiting"
        return 1

    if routingOption not in ("segments", "distance", "time", "scenic"):
        print "Please enter valid routing option...exiting"
        return 1

    if routingAlgo not in ("bfs", "dfs", "ids", "astar"):
        print "Please enter valid routing algorithm...exiting"
        return 1

    if startCity == endCity:
        print "Please enter different start and end city...exiting"
        return 1

#BFS
def bfs(graph, startCity, endCity):
    queue = [(startCity, 0, 0)] #(CityName, MilesSoFar, TimeSoFar)
    time = 0.0
    totTime = 0.0
    relation = {}
    relation[startCity] = ("",0,0,0)
    while queue:
        vertex, miles, totTime = queue.pop(0)
        for next in graph[vertex]:
            if next[0] in relation:
                continue
            relation[next[0]] = (vertex,next[1],next[2],next[3])
            time = float(next[1]) / float(next[2])
            if next[0] == endCity:
                return (relation, int(next[1]) + miles, totTime + time)
            else:
                queue.append((next[0], int(next[1]) + miles, totTime + time))

    return ("",0,-1)

def dfs(graph, startCity, endCity):
    queue = [(startCity, 0, 0)]
    time = 0.0
    totTime = 0.0
    relation = {}
    relation[startCity] = ("",0,0,0)
    while queue:
        vertex, miles, totTime = queue.pop()
        for next in graph[vertex]:
            if next[0] in relation:
                continue
            relation[next[0]] = (vertex,next[1],next[2],next[3])
            time = float(next[1]) / float(next[2])
            if next[0] == endCity:
                return (relation, int(next[1]) + miles, totTime + time)
            else:
                queue.append((next[0], int(next[1]) + miles, totTime + time))

    return ("",0,-1)

def idsdfs(graph, startCity, endCity, limit):
    queue = [(startCity, 0, 0, 0)]
    time = 0.0
    totTime = 0.0
    relation = {}
    relation[startCity] = ("",0,0,0,0)
    visited = {}
    visited[startCity] = 0

    while queue:
        vertex, miles, totTime, level = queue.pop()
        level += 1
        for next in graph[vertex]:

            if next[0] in visited and visited[next[0]] <= level:
                    continue

            visited[next[0]] = level
            relation[next[0]] = (vertex,next[1],next[2],next[3])
            time = float(next[1]) / float(next[2])

            if next[0] == endCity:
                return (relation, int(next[1]) + miles, totTime + time, level)
            elif level < limit:
                queue.append((next[0], int(next[1]) + miles, totTime + time, level))

    return ("",0,0,-1)

#Took code for below haversine function from stackoverflow
from math import radians, cos, sin, asin, sqrt
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 3956 #6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

from heapq import heappush, heappop
def astar(graph, gps, startCity, endCity, minSegment, overallDistance, overallTime):
    heap = []
    heappush(heap, (0, startCity, 0, 0, 0))
    time = 0.0
    totTime = 0.0
    relation = {}
    relation[startCity] = ("",0,0,0)
    visited = []
    visited.append(startCity)
    avgSpeed = overallDistance / overallTime
    maxSpeed = 70
    priorValue = 0
    segment = {}
    segment[startCity] = 0

    if endCity not in gps:
        count = 0
        lat = 0.0
        lon = 0.0
        for city in graph[endCity]:
            if city[0] in gps:
                cor = gps[city[0]]
                count += 1
                lat += float(cor[0])
                lon += float(cor[1])
        if count > 0:
            lat = lat / count
            lon = lon / count
        goalCor = [lat,lon]
    else:
        goalCor = gps[endCity]

    while heap:

        priority, vertex, miles, totTime, seg = heappop(heap)
        neighbor = graph[vertex]
        for next in neighbor:

            if next[0] in visited:
                continue

            visited.append(next[0])

            if next[0] not in relation:
                relation[next[0]] = (vertex,next[1],next[2],next[3])
                segment[next[0]] = segment[vertex] + 1

            time = float(next[1]) / float(next[2])
            totMiles = int(next[1]) + miles

            if next[0] == endCity:
                return (relation, totMiles, totTime + time)

            if(next[0] not in gps):
                for city in graph[next[0]]:
                    if city[0] not in relation:
                        t1 =  float(next[1]) / float(next[2])
                        t2 = float(city[1]) / float(city[2])
                        avgs =( int(next[1]) + int(city[1]) ) / (t1 + t2)
                        neighbor += [[city[0], int(next[1]) + int(city[1]), avgs ,city[3]]]
                        relation[city[0]] = (next[0], city[1], city[2], city[3])
                        segment[city[0]] = segment[next[0]] + 1
            else:
                currCordinates = gps[next[0]]
                h = haversine(float(currCordinates[0]),float(currCordinates[1]),float(goalCor[0]),float(goalCor[1]))

                if routingOption == 'segments':
                    priorValue = segment[next[0]] + (minSegment / h)
                elif routingOption == 'distance':
                    priorValue = totMiles + h
                elif routingOption == 'time':
                    priorValue = totTime + time + ( h / maxSpeed )
                elif routingOption == 'scenic':
                    if( int(next[2]) > 54 ):
                        priorValue = totMiles + h * ( int (next[2])  - 54)
                    else:
                        priorValue = totMiles + h 

                heappush(heap,(priorValue, next[0] , totMiles, totTime + time, seg))

    return ("",0,0,-1)


def printSolution(solution,startCity,endCity):

    relation = solution[0]
    stack = []
    print "\nPlease follow the below path to reach the %s from %s\n" %(endCity, startCity)
    nextCity = endCity
    while(nextCity != startCity):
        #print nextCity, relation[nextCity]
        value = relation[nextCity]
        stack.append((nextCity,value[1],value[2],value[3]));
        nextCity = value[0]

    print ('-' * 120)
    print "  StartCity\t\t\t\tEndCity\t\t\t   Distance\tSpeed\t    Time\t  Via"
    print "\t\t\t\t\t\t\t\t    (miles)   (miles/hr)    (min)"
    print ('-' * 120)
    print ""
    prev = startCity
    output=[]
    while stack: 
        next = stack.pop()
        time = str(int(round(60 * float(next[1])/float(next[2]))))
        print "%s %s %s %s %s %s" %( prev.ljust(35),next[0].ljust(35),next[1].ljust(10),next[2].ljust(10),time.ljust(10),next[3])
        output.append(next[0]);
        prev = next[0]
    print ""
    print ('-' * 120)

    print "\nMachine Readable format\n"
    print solution[1],
    print solution[2],
    print startCity,
    for i in output:
        print i,
        
#Starting Main
if(readArg()):
    sys.exit(0)

startTime = time.time()
graph, minSegment, overallDistance, overallTime = readRoadSegments()
gps = readCityGps()
if(checkInput(graph)):
    sys.exit(0)
endTime = time.time()

#BFS Algo
if(routingAlgo == "bfs"):
    startTime = time.time()
    solution = bfs(graph, startCity, endCity)
    endTime = time.time()
    if(solution[2] == -1):
        print "\nNo route exist between %s and %s. Please check for flights" %(startCity, endCity)
    else:
        print "Time taken to find the path using BFS=",(endTime - startTime),"sec"
        printSolution(solution,startCity,endCity)

elif(routingAlgo == "dfs"):
    startTime = time.time()
    solution = dfs(graph, startCity, endCity)
    endTime = time.time()
    if(solution[2] == -1):
        print "\nNo route exist between %s and %s. Please check for flights" %(startCity, endCity)
    else:
        print "Time taken to find the path using DFS=",(endTime - startTime),"sec"
        printSolution(solution,startCity,endCity)

elif(routingAlgo == "ids"):
    level = -1 
    limit = 0
    startTime = time.time()
    while level == -1:
        limit += 1
        solution = idsdfs(graph, startCity, endCity, limit)
        level = solution[3]
    endTime = time.time()
    print "Time taken to find the path using IDS=",(endTime - startTime),"sec"
    printSolution(solution,startCity,endCity)

elif(routingAlgo == "astar"):
    startTime = time.time()
    solution = astar(graph, gps, startCity, endCity, minSegment, overallDistance, overallTime)
    endTime = time.time()
    print "Time taken to find the path using Astar=",(endTime - startTime),"sec"
    printSolution(solution,startCity,endCity)

sys.exit(0)
