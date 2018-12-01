# Magic Square
N=3
nums = [x + 1 for x in range(0,N*N)]
avg = sum(nums) / N

def search():
    # N = 3 固定の場合
    rows = [[a,b,c] for a in range(1,10) for b in range (1,10) for c in range(1,10) if a != b and b != c and c != a and a + b + c == 15 ]
    cols = [[x,y,z] for x in rows for y in rows for z in rows 
                    if set(x + y + z) == set(range (1,10))
                    and x[0] + y[0] + z[0] == 15
                    and x[1] + y[1] + z[1] == 15
                    and x[2] + y[2] + z[2] == 15
                    and x[0] + y[1] + z[2] == 15
                    and x[2] + y[1] + z[0] == 15]
    return cols
    
results = search()

print(results)
print(len(results))