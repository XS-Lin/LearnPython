# standard
# 8 Queens
import copy

statusTree = []
currentNode = []
results = []

n = 8
chess = [(x, y) for x in range(0, n) for y in range(0, n)]

def checkNode(node):
    isError = False
    if len(node) <= 1:
        return isError
    for q1, q2 in [(q1, q2) for q1 in node for q2 in node if q1 != q2]:
        isError = isError \
            or q1[0] - q2[0] == 0 \
            or q1[1] - q2[1] == 0 \
            or q1[0] - q2[0] == q1[1] - q2[1]
    return isError

def tryAddQueen(node):
    count = len(node)
    for q in chess:
        node.append(q)
        if checkNode(currentNode):
            node.remove(q)
        elif any([set(node) == set(x) for x in statusTree]):
            node.remove(q)
    return len(node) == count + 1
            

print(tryAddQueen(currentNode))
print(currentNode)
# 7個設定できたが。。
