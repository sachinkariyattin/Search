import sys

#### This is to read the file on the server####
def read_input(inputFile):
    #file=open("C:\Third Sem\Artifical Intelligence\Assignmets_data\input.txt")
    file = open(inputFile)
    lines=[line.strip() for line in file.readlines()]
    data=[]
    finald=[]
    codns=[[0,0],[0,1],[0,2],[0,3],[1,0],[1,1],[1,2],[1,3],[2,0],[2,1],[2,2],[2,3],[3,0],[3,1],[3,2],[3,3]]
    for i in lines:
        a=i.split()
        data.append(a)

    for i in data:
        for j in i:
            finald.append(int(j))
    dict={}
    for i in range (0,16):
        dict[int(finald[i])]=codns[i]
    dict['it']=0
    return dict


#is_solve function: This function takes the initial board and returns whether it is solvable or not.

def is_solve(board):
    cal = []
    zero = dict(board)
    zr = zero[0]

    old = board.items()
    look = [[(0, 0), 1], [(0, 1), 2], [(0, 2), 3], [(0, 3), 4], [(1, 0), 5], [(1, 1), 6], [(1, 2), 7], [(1, 3), 8],
            [(2, 0), 9], [(2, 1), 10], [(2, 2), 11], [(2, 3), 12], [(3, 0), 13], [(3, 1), 14], [(3, 2), 15],
            [(3, 3), 0]]

    for i in look:
        i2 = list(i[0])
        for j in old:
            if j[0] not in ("move","it"):
                if j[1] == i2:
                    cal.append(j[0])

    count = zr[0] + 1
    for i in cal:
        if i != 0:
            for j in cal:
                if j != 0:
                    if (i > j and cal.index(i) < cal.index(j)):
                        count += 1
    #print(count)
    zr = board[0]
    if (count % 2 == 0):
        return ([(0, board, '')])
    else:
        print ("Not solvable please input another board")
        return "Not solvable please input another board"

#Heuristic function: The heuristic functions take the input from the successor function and
#then calculate the cost which includes two parts as in the A* does

def heuristic(board):
    cost1=0
    pz = dict(board)
    goal = {1: [0, 0], 2: [0, 1], 3: [0, 2], 4: [0, 3], 5: [1, 0], 6: [1, 1], 7: [1, 2], 8: [1, 3], 9: [2, 0],
            10: [2, 1], 11: [2, 2], 12: [2, 3], 13: [3, 0], 14: [3, 1], 15: [3, 2], 0: [3, 3]}
    # pz=[(1,[0,0]),(2,[0,1]),(3,[0,2]),(4,[0,3]),(5,[1,0]),(6,[1,1]),(7,[1,2]),(8,[1,3])]
    mand = 0
    # for k,v in pz.items()
    for i in range(1, 16):
        dgt = i
        x, y = pz.get(dgt)
        nx, ny = goal.get(dgt)
        #if x != nx or y != ny:
        xdist=abs(x-nx)
        ydist=abs(y-ny)
        if xdist==3:
            xdist=1
        if ydist==3:
            ydist=2
        mand =(ydist + xdist)
        cost1 += mand
    pz['it']=pz.get('it')+1
    return (cost1, pz)

#is_goal function: Check whether the board passed to it is the final or goal state

def is_goal(board):
    goal = {1: [0, 0], 2: [0, 1], 3: [0, 2], 4: [0, 3], 5: [1, 0], 6: [1, 1], 7: [1, 2], 8: [1, 3], 9: [2, 0],
            10: [2, 1], 11: [2, 2], 12: [2, 3], 13: [3, 0], 14: [3, 1], 15: [3, 2], 0: [3, 3]}
    Flag1=True
    Flag2=True
    for i in range(1,16):
        if board[i]==goal[i]:
            Flag1=True
        else:
            Flag2=False
    if Flag1==Flag2:
        return True
    else:
        return False

#Successor(movement): Takes the board and make the legal move and passes
#the information to the heuristic function which is the cost function

def successor(board):
    listoflist = []
    origin = dict(board[1])
    ocost = int(board[0])
    dirs = board[2][:]
    count = int(0)
    z = list(origin[0])
    x = int(z[0])
    y = int(z[1])
    #     y=z.pop()
    #     x=z.pop()
    x1 = int(x - 1)
    while (x1 >= 0 and count < 1):
        for k, v in origin.items():
            if v == [x1, y]:
                origin[k] = [x, y]
                origin[0] = [x1, y]
                dirs=dirs+"D"
                cost = heuristic(origin)
                listoflist.append((cost[1]['it'] + cost[0], cost[1], dirs))
                count += 1
                break
    origin = dict(board[1])
    ocost = int(board[0])
    dirs = board[2][:]
    count = int(0)
    z = list(origin[0])
    x = int(z[0])
    y = int(z[1])
    x1 = int(x + 1)
    while (x1 <= 3 and count < 1):
        for k, v in origin.items():
            if v == [x1, y]:
                origin[k] = [x, y]
                origin[0] = [x1, y]
                dirs = dirs + "U"
                # x1 = x1 - 1
                cost = heuristic(origin)
                listoflist.append((cost[1]['it'] + cost[0], cost[1], dirs))
                # x1 = x1 - 1
                count += 1
                break
    origin = dict(board[1])
    ocost = int(board[0])
    # ndirs=int(board[2][0])
    dirs = board[2][:]
    # ocost = one
    count = int(0)
    z = list(origin[0])
    x = int(z[0])
    y = int(z[1])
    y1 = int(y + 1)
    while (y1 <= 3 and count < 1):
        for k, v in origin.items():
            if v == [x, y1]:
                origin[k] = [x, y]
                origin[0] = [x, y1]
                dirs = dirs + "L"
                cost = heuristic(origin)
                listoflist.append((cost[1]['it'] + cost[0], cost[1], dirs))
                count += 1
                break

    origin = dict(board[1])
    ocost = int(board[0])
    dirs = board[2][:]
    count = int(0)
    z = list(origin[0])
    x = int(z[0])
    y = int(z[1])
    y1 = int(y - 1)
    while (y1 >= 0 and count < 1):
        for k, v in origin.items():
            if v == [x, y1]:
                origin[k] = [x, y]
                origin[0] = [x, y1]
                dirs = dirs + "R"
                cost = heuristic(origin)
                listoflist.append((cost[1]['it'] + cost[0], cost[1], dirs))
                # x1 = x1 - 1
                count += 1
                break
    origin = dict(board[1])
    ocost = int(board[0])
    dirs = board[2][:]
    z = list(origin[0])
    x = int(z[0])
    y = int(z[1])
    if [x, y] == [0, 0]:
        for k, v in origin.items():
            if v == [3,0]:
                origin[0] = v
                origin[k] = [x, y]
                dirs=dirs+"D"
                cost = heuristic(origin)
                listoflist.append((cost[1]['it'] + cost[0], cost[1], dirs))
                origin = dict(board[1])
                dirs = board[2][:]
            elif v == [0,3]:
                origin[0] = v
                origin[k] = [x, y]
                dirs=dirs+"R"
                cost = heuristic(origin)
                listoflist.append((cost[1]['it'] + cost[0], cost[1], dirs))
                origin = dict(board[1])
                dirs = board[2][:]
    elif [x, y] == [3,0]:
        origin = dict(board[1])
        for k, v in origin.items():
            if v == [0,0]:
                origin[0] = v
                origin[k] = [x, y]
                dirs=dirs+"U"
                cost = heuristic(origin)
                listoflist.append((cost[1]['it'] + cost[0], cost[1], dirs))
                origin = dict(board[1])
                dirs = board[2][:]
            elif v == [3,3]:
                origin[0] = v
                origin[k] = [x, y]
                dirs=dirs+"R"
                cost = heuristic(origin)
                listoflist.append((cost[1]['it'] + cost[0], cost[1], dirs))
                origin = dict(board[1])
                dirs = board[2][:]
    elif [x, y] == [3,3]:
        origin = dict(board[1])
        for k, v in origin.items():
            if v == [0,3]:
                origin[0] = v
                origin[k] = [x, y]
                dirs=dirs+"U"
                cost = heuristic(origin)
                listoflist.append((cost[1]['it'] + cost[0], cost[1], dirs))
                origin = dict(board[1])
                dirs = board[2][:]
            elif v == [3,0]:
                origin[0] = v
                origin[k] = [x, y]
                dirs=dirs+"L"
                cost = heuristic(origin)
                listoflist.append((cost[1]['it'] + cost[0], cost[1], dirs))
                origin = dict(board[1])
                dirs = board[2][:]
    elif [x, y] == [0,3]:
        origin = dict(board[1])
        for k, v in origin.items():
            if v == [0,0]:
                origin[0] = v
                origin[k] = [x, y]
                dirs=dirs+"L"
                cost = heuristic(origin)
                listoflist.append((cost[1]['it'] + cost[0], cost[1], dirs))
                origin = dict(board[1])
                dirs = board[2][:]
            if v == [3,3]:
                origin[0] = v
                origin[k] = [x, y]
                dirs=dirs+"D"
                cost = heuristic(origin)
                listoflist.append((cost[1]['it'] + cost[0], cost[1], dirs))
                origin = dict(board[1])
                dirs = board[2][:]

    elif [x, y] == [0,1]:
        origin = dict(board[1])
        for k, v in origin.items():
            if v == [3,1]:
                origin[0] = v
                origin[k] = [x, y]
                dirs=dirs+"D"
                cost = heuristic(origin)
                listoflist.append((cost[1]['it'] + cost[0], cost[1], dirs))
                origin = dict(board[1])
                dirs = board[2][:]
    elif [x, y] == [3, 1]:
        origin = dict(board[1])
        for k, v in origin.items():
            if v == [0,1]:
                origin[0] = v
                origin[k] = [x, y]
                dirs=dirs+"U"
                cost = heuristic(origin)
                listoflist.append((cost[1]['it'] + cost[0], cost[1], dirs))
                origin = dict(board[1])
                dirs = board[2][:]

    elif [x, y] == [0,2]:
        origin = dict(board[1])
        for k, v in origin.items():
            if v == [3,2]:
                origin[0] = v
                origin[k] = [x, y]
                dirs=dirs+"D"
                cost = heuristic(origin)
                listoflist.append((cost[1]['it'] + cost[0], cost[1], dirs))
                origin = dict(board[1])
                dirs = board[2][:]
    elif [x, y] == [3, 2]:
        origin = dict(board[1])
        for k, v in origin.items():
            if v == [0,2]:
                origin[0] = v
                origin[k] = [x, y]
                dirs=dirs+"U"
                cost = heuristic(origin)
                listoflist.append((cost[1]['it'] + cost[0], cost[1], dirs))
                origin = dict(board[1])
                dirs = board[2][:]
    elif [x, y] == [1,0]:
        origin = dict(board[1])
        for k, v in origin.items():
            if v == [1,3]:
                origin[0] = v
                origin[k] = [x, y]
                dirs=dirs+"R"
                cost = heuristic(origin)
                listoflist.append((cost[1]['it'] + cost[0], cost[1], dirs))
                origin = dict(board[1])
                dirs = board[2][:]
    elif [x, y] == [1,3]:
        origin = dict(board[1])
        for k, v in origin.items():
            if v == [1,0]:
                origin[0] = v
                origin[k] = [x, y]
                dirs=dirs+"L"
                # x1 = x1 - 1
                cost = heuristic(origin)
                listoflist.append((cost[1]['it'] + cost[0], cost[1], dirs))
                origin = dict(board[1])
                dirs = board[2][:]
    elif [x, y] == [2,0]:
        origin = dict(board[1])
        for k, v in origin.items():
            if v == [2,3]:
                origin[0] = v
                origin[k] = [x, y]
                dirs=dirs+"R"
                # x1 = x1 - 1
                cost = heuristic(origin)
                listoflist.append((cost[1]['it'] + cost[0], cost[1], dirs))
                origin = dict(board[1])
                dirs = board[2][:]
    elif [x, y] == [2,3]:
        origin = dict(board[1])
        for k, v in origin.items():
            if v == [2,0]:
                origin[0] = v
                origin[k] = [x, y]
                dirs=dirs+"L"
                cost = heuristic(origin)
                listoflist.append((cost[1]['it'] + cost[0], cost[1], dirs))
                origin = dict(board[1])
                dirs = board[2][:]
    return listoflist

#Solver: Calls other functions and solves 15 puzzle
def solver(initial):
    count=1
    # is_first = 0
    visited = []
    fringe = []
    if is_goal(initial):
        print("goal")
        return (initial)
    else:
        a = is_solve(initial)
        if a != "Not solvable please input another board":
            fringe = fringe + a
    while (len(fringe) > 0 and count==1):
        # count=count+1
        s, d, r = fringe.pop(0)
        if is_goal(d):
            counter=0
            for i in r:
                counter+=1
                ll='[{0}-{1}]'.format(i,counter)
                print (ll,end='')
            count=count+1
        else:
            if d not in visited:
                visited.append(d)
                # if is_solve(d):
                j = (s, d, r)
                m = successor(j)
                fringe = fringe + m
                fringe.sort(key=lambda x: x[0])

    if count!=2:
        print("Fringe Empty")

inputFile = sys.argv[1]
# calling the solver
solver(read_input(inputFile))
####Note: I am first time programmer apologies for the long code##########



