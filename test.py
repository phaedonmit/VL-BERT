# def check(*args):
#     [one, two, three] = args
#     print(len(args))
#     print(one)
import torch

if __name__ == "__main__":
    # check('this', 'is', 'the')

    multi = torch.zeros((5,8))
    print(multi)
    x = [1.,2.,3.,4.,5.]

    for i in range(5):
        mlm = torch.zeros((8,))+x[i]
        multi[i,:] = mlm
        print(multi)