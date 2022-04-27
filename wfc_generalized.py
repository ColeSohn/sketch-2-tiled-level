import pdb 
import numpy as np
import sys
import matplotlib.pyplot as plt

def make_constraint_array(p):
    #create array and initialize total counts
    counts = np.bincount(p.flatten())
    tile_types = counts.shape[0]
    c_a = np.zeros((tile_types,tile_types*4))
    #pdb.set_trace()

    #north constraints
    for i in range(1, p.shape[0]):
        for j in range(p.shape[1]):
            offset = 0
            x = p[i][j]
            x_off = p[i-1][j]
            c_a[x][tile_types*offset+x_off]+=1
    
    #south constraints
    for i in range(p.shape[0]-1):
        for j in range(p.shape[1]):
            offset = 2
            x = p[i][j]
            x_off = p[i+1][j]
            c_a[x][tile_types*offset+x_off]+=1
    
    #west constraints
    for i in range(p.shape[0]):
        for j in range(1, p.shape[1]):
            offset = 3
            x = p[i][j]
            x_off = p[i][j-1]
            c_a[x][tile_types*offset+x_off]+=1

    #east constraints
    for i in range(p.shape[0]):
        for j in range(p.shape[1]-1):
            offset = 1
            x = p[i][j]
            x_off = p[i][j+1]
            c_a[x][tile_types*offset+x_off]+=1
    c_a[c_a>1]=1
    return c_a, counts

def find_neighbor_direction(me, neighbor, game_board):
    return np.where(game_board[int(me)]==neighbor)[0][0]

#returns new constraints
def update_state(face, states, game_board, constraints):
    num_tiles = constraints.shape[0]
    #pdb.set_trace()
    out=states[face]
    #get constraints from all surrounding tiles
    for i in range(4):
        nb_constraints = np.zeros_like(states[0])
        tile_idx = game_board[face][i] #Get connected tile
        if(tile_idx==-1):
            nb_constraints = np.ones_like(states[0])
        else:
            for j, s in enumerate(states[int(tile_idx)]):
                #find current tile's idx in neighbor's piece
                if(s):
                    #dir = find_neighbor_direction(tile_idx, face, game_board)
                    if(i==0):
                        nb_constraints=np.logical_or(nb_constraints, constraints[j][num_tiles*2:num_tiles*3])
                    elif(i==1):
                        nb_constraints=np.logical_or(nb_constraints, constraints[j][num_tiles*3:num_tiles*4])
                    elif(i==2):
                        nb_constraints=np.logical_or(nb_constraints, constraints[j][num_tiles*0:num_tiles*1])
                    elif(i==3):
                        nb_constraints=np.logical_or(nb_constraints, constraints[j][num_tiles*1:num_tiles*2])
                    #nb_constraints=np.logical_or(nb_constraints, constraints[j][num_tiles*dir:num_tiles*(dir+1)])
        out=out*nb_constraints
    return out

def propagate(face_id, states, game_board, constraints):
    og_face = face_id
    q = [face_id]
    while(len(q)):
        #if(og_face == 4):
        #    pdb.set_trace()
        #pdb.set_trace()
        #Step 1: Check surrounding neighbors for necessary constraint changes. If necessary, change constraints and add to queue
        #1a: get neighbors
        neighs = game_board[q.pop(0)]
        for n in neighs:
            n = int(n)
            #pdb.set_trace()
            if(n!=-1):
                old_state = states[n]
                #Skip collapsed neighbors
                if(np.sum(old_state)==1):
                    continue
                #New constraints. If no change, don't append
                new_state = update_state(n, states, game_board, constraints)
                if(not np.array_equal(new_state, old_state)):
                    q.append(n)
                    states[n]=new_state
    return states

#Inputs: 
#   1. game board array(nx4) - tiles x connected tiles (-1 for no connection) [upper, right, lower, left]
#   2. patterns array (nxm) - any 2d array representing tile constraints/probs
def wfc(game_board, patterns):
    constraints, total_counts = make_constraint_array(patterns)
    print("Constraints", constraints)
    states = np.ones((game_board.shape[0], constraints.shape[0]))

    cnt = 0
    print(states)
    while(True):
        cnt+=1
        entropies = np.sum(states, axis=1)
        if not np.all(entropies):
            #print(entropies)
            return []
        if np.all(entropies == 1):
            return states
        if(cnt>10000):
            print("ERROR: inf loop")
            break

        #Step 1: get face with lowest nonzero entropy
        entropies[entropies == 1] = np.inf
        current_face = np.argsort(entropies)[0]
        

        #Step 2: Find valid tile to place on face (stochastically by tile count in patter input)
        collapse_probs=states[current_face]*total_counts
        collapse_probs/=np.sum(collapse_probs)
        choice = np.random.choice(np.arange(0, len(collapse_probs), 1, dtype=np.int32), p=collapse_probs)

        print("Choosing for space", str(current_face)+".", "Chose tile", choice)

        #Step 3: Assign Face
        new_state = np.zeros(total_counts.shape[0])
        new_state[choice]=1
        states[current_face]=new_state
        #print("States Iteration:", cnt,":", states)
        #Step 4 (Big): Propagate Changes (ie. eliminate impossible tiles on all connected faces recursively)
        states=propagate(current_face, states, game_board, constraints)
        print("States", states)
    return states

def game_board_square(shape):
    x = shape[1]
    y = shape[0]

    game_board=np.zeros((x*y, 4))
    for i in range(y):
        for j in range(x):
            c = y*i+j
            if(i==0):
                game_board[c][0]=-1
            else:
                game_board[c][0]=c-x

            if(j==x-1):
                game_board[c][1]=-1
            else:
                game_board[c][1]=c+1
            
            if(i==y-1):
                game_board[c][2]=-1
            else:
                game_board[c][2]=c+x
            
            if(j==0):
                game_board[c][3]=-1
            else:
                game_board[c][3]=c-1
    return game_board

def grid_out_to_plot(states, shape, patterns, path): 
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('WFC')

    data = np.reshape(states, shape).astype(float)/float(np.amax(patterns))
    ax1.imshow(patterns, interpolation='nearest')
    ax2.imshow(data, interpolation='nearest')
    #plt.savefig(path)
    plt.show()


if __name__ == "__main__":
    np.random.seed(30)
    board_shape = (50, 50)
    game_board = game_board_square(board_shape)
    #print(game_board)
    #game_board = np.array([[-1, 1, 2, -1],[-1, -1, 3, 0], [0, 3, -1, -1], [1, -1, -1, 2]])
    patterns = np.zeros((5, 4), dtype=np.int32)
    patterns[1][1]=1
    patterns[1][2]=2

    A = []
    cnt = 0
    out = -1
    while(True):
        A = wfc(game_board, patterns)
        if(not len(A)):
            print("contradiction: retrying")
            cnt+=1
            if(cnt>10):
                print("giving up")
                break
        else:
            out = np.argmax(A, axis=1)
            grid_out_to_plot(out, board_shape, patterns, "G:\My Drive\CS191W\plots\out.png")
            break
    
    print(out)



