def sockMerchant(n, ar)
    #converting list into set so no repetely count in the loop.
    s = set(ar)
    c=0
    for item in s:
        c = ar.count(item)//2 + c
    print(c)