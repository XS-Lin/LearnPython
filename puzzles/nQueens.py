# 8 Queens
N = 4
CHESS = [(x, y) for x in range(0, N) for y in range(0, N)]

class State:
    def __init__(self):
        self.value = []

    def rotateDeg90(self):
        ''' 90°回転変換 '''
        result = []
        for q in self.value:
            result.append((N-1-q[1], q[0]))
        self.value = result

    def reflectY(self):
        ''' 縦軸対称変換 '''
        result = []
        for q in self.value:
            result.append((N-1-q[0], q[1]))
        self.value = result

    def isEqual(self, state):
        ''' 盤面状態が同様か判定(0°,90°,180°,270°回転と縦軸対称変換を含む) '''
        selfCopy = self.copy()
        stateCopy = state.copy()
        stateCopy.reflectY()
        if set(selfCopy.value) == set(state.value) or set(selfCopy.value) == set(stateCopy.value):
            return True
        selfCopy.rotateDeg90()
        if set(selfCopy.value) == set(state.value) or set(selfCopy.value) == set(stateCopy.value):
            return True
        selfCopy.rotateDeg90()
        if set(selfCopy.value) == set(state.value) or set(selfCopy.value) == set(stateCopy.value):
            return True
        selfCopy.rotateDeg90()
        if set(selfCopy.value) == set(state.value) or set(selfCopy.value) == set(stateCopy.value):
            return True
        return False

    def validMovesData(self):
        ''' 盤面の状態で次の一手のすべての可能値'''
        return [q for q in CHESS if all([
                    q != s and q[0] != s[0] and q[1] != s[1] 
                    and abs(q[0] - s[0]) != abs(q[1] - s[1])
                    for s in self.value
                ])]

    def copy(self):
        ''' 盤面の状態を複製'''
        state = State()
        state.value = [s for s in self.value]
        return state

    def storeData(self, q):
        ''' 手を打つ '''
        self.value.append(q)

results = []

def search():
    stack = [State()]
    finished = set()

    while len(stack) > 0:
        n = stack.pop()
        finished.add(n.copy())

        for q in n.validMovesData():
            nextState = n.copy()
            nextState.storeData(q)

            if not any(nextState.isEqual(x) for x in finished):
                if len(nextState.value) == N:
                    if not any(nextState.isEqual(x) for x in results):
                       results.append(nextState)
                    continue
                stack.append(nextState)

search()
for r in results:
    print(r.value)
print(len(results))
