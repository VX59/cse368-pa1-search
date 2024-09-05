# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
from game import Directions
import util
from util import Stack, Queue, PriorityQueue, PriorityQueueWithFunction
import math

class SearchTreeNode:
    def __init__(self, state):
        self.state = state
        self.parent = None

class GenericSearchAlgorithm:
    def __init__(self,method_name,problem):
        method =  globals()[method_name]
        self.method_name = method_name
        self.method = method()
        self.problem = problem
        self.discoveryTreeRoot = None

    # first find distance from goal point
    def cost(self, s):
        p = (1,1)
        if p == s: return 0
        df = math.sqrt(pow(s[0]-p[0],2)+pow(s[1]-p[1],2))
        
        p = self.problem.getStartState()
        if p == s: return 0
        ds = math.sqrt(pow(s[0]-p[0],2)+pow(s[1]-p[1],2))

        return (df-ds)

    def __call__(self,s):
        S = SearchTreeNode(s)
        discovered = [s]
        self.discoveryTreeRoot = discovered[0]
        if self.method_name == "PriorityQueue":
            self.method.push(S,self.cost(s))
        elif self.method_name == "PriorityQueueWithFunction":
            pass
        else:
            self.method.push(S)

        self.directions = {'North':Directions.NORTH,
                            'South':Directions.SOUTH,
                            'East':Directions.EAST,
                            'West':Directions.WEST}

        while not self.method.isEmpty():
            V = self.method.pop()
            if V.state == self.problem.getStartState():
                adjacent = self.problem.getSuccessors(V.state)
            else:
                adjacent = self.problem.getSuccessors(V.state[0])
 
            for u in adjacent:
                position = u[0]
                U = SearchTreeNode(u)

                if not position in discovered:
                    discovered.append(position)
                    U.parent = V
                    # update for all search methods
                    if self.method_name == "PriorityQueue":
                        self.method.push(U,self.cost(u[0]))
                    elif self.method_name == "PriorityQueueWithFunction":
                        pass
                    else:
                        self.method.push(U)

                if self.problem.isGoalState(position):
                    A = []
                    while U != None:
                        d = U.state[-2]
                        if type(d) != int:
                            A.insert(0,self.directions[d])         
                        U = U.parent
                    return A

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    s = problem.getStartState()
    dfsSearch = GenericSearchAlgorithm("Stack", problem)
    solution = dfsSearch(s)
    print(solution)
    return solution

def breadthFirstSearch(problem):
    s = problem.getStartState()
    bfsSearch = GenericSearchAlgorithm("Queue", problem)
    solution = bfsSearch(s)
    print(solution)
    return solution

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    s = problem.getStartState()
    ucsSearch = GenericSearchAlgorithm("PriorityQueue", problem)
    solution = ucsSearch(s)
    print(solution)
    return solution

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
