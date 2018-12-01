# Celebrity problem
# 有名人の問題
# N人のグループに有名人が一人いる。その有名人はグループ内に誰も知人がいないが、他のすべての人はその人を知っている。
# ここで「あなたはこの人を知っているか？」という質問だけで、その人を見つける方法を示せ。

import random
N = 9
x = random.randint(0,N)

class Person:
    def __init__(self,id):
        self.id = id
        self.__target = id == x
        
    def knowsOther(self,person):
        if person is None or self == person:
            return "meaningless" # ある人に存在しない人、または本人を知るか聞いても意味がない
        if self.__target:
            return False
        elif person.__target:
            return True
        else:
            return "unknow" # 問題文に記載されていない

group = [Person(id) for id in range(0,N)]

#knowsOtherでx取得
result = []
def search(init,result):
    if len(result) == 1:
        return
    elif len(init) == 1:
        result.append(init.pop())
    else:
        a = init.pop()
        b = init[0]
        if not a.knowsOther(b):
            result.append(a)
        else:
            search(init,result)
        
search(group,result)
print(result[0].id)
print(x)