

def add(a:int, b:int) -> int:
    ret:int = a+b

    breakpoint()

    ret += 69
    
    return ret



if __name__ == '__main__':
    print(add(12,12))