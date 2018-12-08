# Two Jealous Husband
# 二人の嫉妬深い夫
#   二組の夫婦が川を渡らなければならない。舟はあるが一度に二人までしか乗れないものとする。
# さらにややこしいことに、どちらの夫もたいへん嫉妬深いので、自分がいないときに自分の妻と相手の夫が一緒にいることを好まない。
# この条件で川を渡ることはできるか。

initial = [["W1","H1","W2","H2"],[]]
goal = [[],["W1","H1","W2","H2"]]

class Boat:
    def __init__(self,state):
        self.position = 0 # 0 此岸 1 彼岸
        self.state = state
    def carry(self,pilot,passenger = None):
        if self.position == 0:
            self.state[0].remove(pilot)
            if passenger is not None:
                self.state[0].remove(passenger)
            self.state[1].append(pilot)
            if passenger is not None:
                self.state[1].append(passenger)
            self.position = 1
        else:
            self.state[1].remove(pilot)
            if passenger is not None:
                self.state[1].remove(passenger)
            self.state[0].append(pilot)
            if passenger is not None:
                self.state[0].append(passenger)
            self.position = 0

def limit(state):
    return ("W1" in state[0] and "H2" in state[0]) \
        or ("W1" in state[1] and "H2" in state[1]) \
        or ("W2" in state[0] and "H1" in state[0]) \
        or ("W2" in state[1] and "H1" in state[1])

def copyState(state):
    return [state[0].copy()],[state[1].copy()]

def getAllValidState(state,position):
    validState = []

    # 一人移動のパターン
    if position == 0:
        for x in state[0]:
            copyS = copyState(state)
            copyS[0].remove(x)
            copyS[1].append(x)
            if not limit(copyS):
                validState.append(copyS)
    else:
        for x in state[0]:
            copyS = copyState(state)
            copyS[1].remove(x)
            copyS[0].append(x)
            if not limit(copyS):
                validState.append(copyS)

    # 二人移動のパターン
    if position == 0:
        for x in state[0]:
            for y in state[0]:
                if x == y:
                    continue
                copyS = copyState(state)
                copyS[0].remove(x)
                copyS[0].remove(y)
                copyS[1].append(x)
                copyS[1].append(y)
                if not limit(copyS):
                    validState.append(copyS)
    else:
        for x in state[0]:
            for y in state[0]:
                if x == y:
                    continue
                copyS = copyState(state)
                copyS[1].remove(x)
                copyS[1].remove(y)
                copyS[0].append(x)
                copyS[0].append(y)
                if not limit(copyS):
                    validState.append(copyS)
    return validState

resultRoot = []

def search(initial,goal):
    if set(initial[1]) == set(goal[1]):
        return
    result = []
    queue = [copyState(initial)]
    finished = []
    while len(queue) > 0:
        n = queue.pop(0)
        boat = Boat(n)
        for nextState in getAllValidState(boat.state,boat.position):
            if nextState not in finished:
                queue.append(nextState)
            finished.append(nextState)
    return result
    