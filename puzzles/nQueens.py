# 8 Queens
N = 4
CHESS = [(x, y) for x in range(0, N) for y in range(0, N)]

class State:
    def __init__(self):
        self.value = []

    def isEqual(self, state):
        return set(self.value) == set(state.value)

    def validMovesData(self):
        return [q for q in CHESS if all([
                    q != s and q[0] != s[0] and q[1] != s[1] and q[0] - s[0] != q[1] - s[1]
                    for s in self.value
                ])]

    def copy(self):
        state = State()
        state.value = [s for s in self.value]
        return state

    def storeData(self, q):
        self.value.append(q)

results = []
def search():
    stack = [State()]
    finished = set()

    while len(stack) > 0:
        n = stack.pop()
        finished.add(n.copy())
        #print(len(stack))
        #print(len(n.value))
        #print(n.value)
        for q in n.validMovesData():
            nextState = n.copy()
            nextState.storeData(q)
            
            if not any(nextState.isEqual(x) for x in finished):
               if len(nextState.value) == N:
                   results.append(nextState.value)
                   continue
               stack.append(nextState)

search()
for r in results:
    print(r)
print(len(results))
