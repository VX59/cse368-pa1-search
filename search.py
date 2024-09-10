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
import numpy as np

class Node:
    def __init__(self,state,direction, weight):
        self.state = state
        self.direction = direction
        self.weight = weight

class GraphSearch:
    def __init__(self,methodName,problem,heuristic=lambda x,y: 0):
        self.methodName = methodName
        self.problem = problem
        self.heuristic = heuristic
        self.method = globals()[methodName]()
        self.ancestors = {}
    
        self.directions = {'North':Directions.NORTH,
                        'South':Directions.SOUTH,
                        'East':Directions.EAST,
                        'West':Directions.WEST}

    def push(self,item,priority):
        if self.methodName == "PriorityQueue":
            self.method.push(item,priority)
        else:
            self.method.push(item)

    def reconstructPath(self,state):
        path = []
        while state.direction != None:
            path.append(state.direction)
            state = self.ancestors[state]
        path.reverse()
        return path

    def __call__(self,s):
        discovered = [s]

        self.push((None,Node(s,None,0)),0)

        while not self.method.isEmpty():
            prevNode, node = self.method.pop()
            currentState = node.state
            self.ancestors[node] = prevNode
            if self.problem.isGoalState(currentState):
                print("completed")
                return self.reconstructPath(node)

            neighbors = self.problem.getSuccessors(currentState)

            for neighbor, statedirection, unitWeight in neighbors:
                if not neighbor in discovered:
                    newWeight = unitWeight + node.weight
                    self.push((node,Node(neighbor,statedirection,newWeight)), newWeight + self.heuristic(neighbor,self.problem))
                    discovered.append(neighbor)

methods = {
    "bfs":"Queue",
    "dfs":"Stack",
    "ucs":"PriorityQueue",
    "astar":"PriorityQueue"
}

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
    dfsSearch = GraphSearch(methods["dfs"], problem)
    solution = dfsSearch(s)
    print(solution)
    return solution

def breadthFirstSearch(problem):
    s = problem.getStartState()
    bfsSearch = GraphSearch(methods["bfs"], problem)
    solution = bfsSearch(s)
    print(solution)
    return solution

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    s = problem.getStartState()
    ucsSearch = GraphSearch(methods["ucs"], problem)
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
    s = problem.getStartState()
    astarSearch = GraphSearch(methods["astar"], problem, heuristic=heuristic)
    solution = astarSearch(s)
    return solution


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch