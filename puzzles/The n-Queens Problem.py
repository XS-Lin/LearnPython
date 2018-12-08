# The n-Queens Problem
# nクイーン問題
#  n個のクイーンをn*nマスのチェス盤に置いて、どの二つのクイーンも互いに攻撃させないようにせよ。
# すなわち、どの二つのクイーンも同じ行、列あるいは斜め線上にないようにせよ。

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

results1 = []

# 基本的な考え方ですが N > 6 の場合は遅すぎる
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
                    if not any(nextState.isEqual(x) for x in results1):
                       results1.append(nextState)
                    continue
                stack.append(nextState)

results2 = []
# 改善：クラスを使用しないで、オブジェクト生成処理時間を節約(N > 6の場合改善はわずか、実用ではない)
def search2():
    stack = [[]]
    finished = []

    while len(stack) > 0:
        n = stack.pop()
        finished.append([_ for _ in n])
        for x,y in [(x,y) for x,y in CHESS if not any ( x==x0 or y == y0 or abs(x-x0) == abs(y-y0) for x0,y0 in n)]:
            nextS = [_ for _ in n]
            nextS.append((x,y))

            if not any(set(nextS)==set(i) for i in finished):
                if len(nextS) == N:
                    nextS.sort(key = lambda pos:(pos[0],pos[1]))
                    if nextS not in results2:
                        results2.append(nextS)
                    continue
                stack.append(nextS)

results3 = []
# 改善：盤面状態の表記を位置(タプル)より数字変換、オブジェクト処理時間をを短縮(N=8 の時数分間かかる)
def search3():
    stack = [[]]
    finished = []

    while len(stack) > 0:
        n = stack.pop()
        finished.append([_ for _ in n])

        validPos = [_ for _ in range(0,N*N)]
        for pos in range(0,N*N):
            x,y = divmod(pos,N)

            for nPos in n:
                x0,y0 = divmod(nPos,N)
                if x==x0 or y == y0 or abs(x-x0) == abs(y-y0):
                    if pos in validPos:
                        validPos.remove(pos)
                    
        #print(nextPos)
        for pos in validPos:
            nextS = [_ for _ in n]
            nextS.append(pos)
            nextS.sort()
            if nextS not in finished:
                if len(nextS) == N:
                    if nextS not in results3:
                        results3.append(nextS)
                    continue
                stack.append(nextS)

results4=[]
# 改善：盤面状態の判断を2進数で、有効位置取得方法改善(N=8の場合、何分間かかる)
def search4(init = []):
    stack = [init]
    finished = []
    while len(stack) > 0:
        n = stack.pop()
        finished.append([_ for _ in n])

        validPosX = pow(2,N) - 1 #水平状況を記録２進数(１：配置可能 0：エラー)
        validPosY = pow(2,N) - 1 #垂直状況を記録２進数(１：配置可能 0：エラー)
        validPosZ = pow(2,2*N-1) - 1 #斜め(00->NN)状況を記録２進数(１：配置可能 0：エラー)
        validPosXY = pow(2,2*N-1) - 1 #斜め(0N->N0)状況を記録２進数(１：配置可能 0：エラー)
        validPosYX = pow(2,2*N-1) - 1 #斜め(N0->0N)状況を記録２進数(１：配置可能 0：エラー)
        
        for nPos in n:
            x0,y0 = divmod(nPos,N)
            validPosX = validPosX & ~(pow(2,x0))
            validPosY = validPosY & ~(pow(2,y0))
            validPosZ = validPosZ & ~(pow(2,x0 + y0))
            validPosXY = validPosXY & ~(pow(2,x0 - y0 + N))
            validPosYX = validPosYX & ~(pow(2,y0 - x0 + N))
        
        validPos = []
        for pos in range(0,N*N):
            x,y = divmod(pos,N)
            if pow(2,x) == (pow(2,x) & validPosX) \
                and pow(2,y) == (pow(2,y) & validPosY) \
                and pow(2,x + y) == (pow(2,x + y) & validPosZ) \
                and (pow(2,x - y + N) == (pow(2,x - y + N) & validPosXY) or pow(2,y - x + N) == (pow(2,y - x + N) & validPosYX)) :

                if pos not in validPos:
                    validPos.append(pos)
                    
        #print(nextPos)
        for pos in validPos:
            nextS = [_ for _ in n]
            nextS.append(pos)
            nextS.sort()
            if nextS not in finished:
                if len(nextS) == N:
                    if nextS not in results4:
                        results4.append(nextS)
                    continue
                stack.append(nextS)

results5=[]
def search5(init = []):
    stack = [init]
    finished = []
    while len(stack) > 0:
        n = stack.pop()
        finished.append([_ for _ in n])

        validPosX = pow(2,N) - 1 #水平状況を記録２進数(１：配置可能 0：エラー)
        validPosY = pow(2,N) - 1 #垂直状況を記録２進数(１：配置可能 0：エラー)
        validPosZ = pow(2,2*N-1) - 1 #斜め(00->NN)状況を記録２進数(１：配置可能 0：エラー)
        validPosXY = pow(2,2*N-1) - 1 #斜め(0N->N0)状況を記録２進数(１：配置可能 0：エラー)
        validPosYX = pow(2,2*N-1) - 1 #斜め(N0->0N)状況を記録２進数(１：配置可能 0：エラー)
        
        for nPos in n:
            x0,y0 = divmod(nPos,N)
            validPosX = validPosX & ~(pow(2,x0))
            validPosY = validPosY & ~(pow(2,y0))
            validPosZ = validPosZ & ~(pow(2,x0 + y0))
            validPosXY = validPosXY & ~(pow(2,x0 - y0 + N))
            validPosYX = validPosYX & ~(pow(2,y0 - x0 + N))
        
        validPos = []
        for pos in range(0,N*N):
            x,y = divmod(pos,N)
            if 1 & (validPosX >> x) \
                and 1 & (validPosY >> y) \
                and 1 & (validPosZ >> (x + y)) \
                and (1 & (validPosXY >> (x - y + N)) or 1 & ( validPosYX >> (y - x + N))) :

                if pos not in validPos:
                    validPos.append(pos)
                    
        #print(nextPos)
        for pos in validPos:
            nextS = [_ for _ in n]
            nextS.append(pos)
            nextS.sort()
            if nextS not in finished:
                if len(nextS) == N:
                    if nextS not in results5:
                        results5.append(nextS)
                    continue
                stack.append(nextS)

#search()
#for r in results1:
#    print(r.value)
#print(len(results1))

#search2()
#for r in results2:
#    print(r)
#print(len(results2))

#search3()
#for r in results3:
    #print(list([divmod(x,N) for x in r]))
#    print(r)
#print(len(results3))

#search4()
#for r in results4:
    #print(list([divmod(x,N) for x in r]))
#    print(r)
#print(len(results4))

search5()
for r in results5:
    #print(list([divmod(x,N) for x in r]))
    print(r)
print(len(results5))

#print([x for x in results3 if x not in results4])